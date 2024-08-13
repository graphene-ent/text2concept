import os
import time
import json

import asyncio  
import nest_asyncio

import litellm

from tqdm import tqdm

class GPTPipeline:

    def __init__(self,
                 system_prompt:str="",
                 user_prompt:str="'{}'",
                 few_shot_examples:list=[],
                 model_name:str="gemini-1.5-flash",
                 temperature:int=0,
                 as_json:bool=False):
        
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.few_shot_examples = few_shot_examples

        self.model_name = model_name
        self.temperature = temperature

        self.as_json = as_json

    def prompt_chatgpt(self, prompt: str) -> str:

        prompt = self.user_prompt.format(prompt)

        messages = []

        system_prompt = [{"role" : "system", "content" : self.system_prompt}]
        messages.extend(system_prompt)

        if self.few_shot_examples != []:
            examples = [[{"role" : "user", "content" : examples[0]},{"role" : "assistant", "content" : examples[1]}]
                        for examples in self.few_shot_examples]
            examples = [arr for sublist in examples for arr in sublist]
            messages.extend(examples)

        prompt = [{"role" : "user", "content" : prompt}]
        messages.extend(prompt)
  
        try:  
            completion = litellm.completion(model=f"openai/{self.model_name}-dev", 
                                            messages=messages,
                                            temperature=self.temperature,
                                            base_url=os.getenv("LITELLM_BASE_URL"),
                                            api_key=os.getenv("LITELLM_API_KEY"),
                                            user=f"mr-{os.getenv('ENV')}-concept-extraction"
                                            )
            return completion['choices'][0]['message']['content']
        
        except Exception as e:
            print(f"Exception : {e}")
            return json.dumps("")
    
    def get_output(self,
                   texts:list) -> str:
        
        time.sleep(1.5) #to avoid being rate limited

        output = self.prompt_chatgpt(prompt=texts)

        if not self.as_json:
            return output

        try:
            output = json.loads(output)
        except ValueError as v:
            print("Error : {}".format(v))
            output = self.prompt_chatgpt(prompt=texts)
            output = json.loads(output)

        return output

class AsyncLLMPipeline:
    def __init__(self, model:str=None):
        self.model = model

    async def predict(self,
                      user_prompt,
                      max_retries:int=5):
        raise NotImplementedError

    async def model_response(self,
                             user_prompts):
        
        if not isinstance(user_prompts,list):
            user_prompts = [user_prompts]

        tasks = [self.predict(user_prompt) for user_prompt in user_prompts]
            
        responses = await asyncio.gather(*tasks)
        timeout_or_incorrect_resp = 0 #count for how many requests timed out or have incorrect responses
        
        decoded_responses = []
        for resp in responses:

            try:
                decoded_response = resp['choices'][0]['message']['content']
            
            except TypeError as te:
                print(resp)
                print(te)
                timeout_or_incorrect_resp += 1
                decoded_response = 'indeterminate'
                
            except Exception as e:
                print(resp, e)
                decoded_response = 'indeterminate'
            finally:
                decoded_responses.append(decoded_response)

        print(f"{timeout_or_incorrect_resp} requests out of {len(tasks)} requests either timed out or returned non-parseable outputs ...")
            
        return decoded_responses

    def batch_predict(self,
                      user_prompts):
        
        batched_prompts = [user_prompts[idx : idx+50]
                           for idx in range(0, len(user_prompts), 50)]
        
        outputs = []
        for batch in tqdm(batched_prompts):
            batch_output = asyncio.run(self.model_response(batch))
            outputs.extend(batch_output)

            if self.model in ["gemini-ultra"]:
                time.sleep(50) #sleep for a WAY longer time due to resource_exhaustion limits on the gemini API

            time.sleep(1) #sleep for a second after each batch is processed!

        return outputs

nest_asyncio.apply()

class AsyncLLMPipelineBuilder(AsyncLLMPipeline):

    def __init__(self,
                 system_prompt:str,
                 few_shot_examples:list=[],
                 model:str="gemini-1.5-flash",
                 max_timeout_per_request:int=15):

        self.system_prompt = system_prompt
        self.few_shot_examples = few_shot_examples
        self.model = model

        self.max_timeout_per_request = max_timeout_per_request

        #if model name has 'gpt-' in it, add a '-dev' suffix to it
        if 'gpt-' in self.model:
            self.model = self.model + '-dev'

        print(f'changed model name to {self.model}\n')

        print(f'Using model : {self.model}')

        super().__init__(model=self.model)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        print('Exit called ... cleaning up')

        self.model = None

        print('Cleanup complete!\n')

        return True

    async def predict(self,
                      user_prompt,
                      max_retries:int=5):

        retries = 0
        backoff_factor = 2
        min_sleep_time = 3
        
        time.sleep(0.3) #to avoid getting rate limited immediately

        messages = []

        system_prompt = [{"role" : "system", "content" : self.system_prompt}]
        messages.extend(system_prompt)

        if self.few_shot_examples != []:
            examples = [[{"role" : "user", "content" : examples[0]},{"role" : "assistant", "content" : examples[1]}]
                        for examples in self.few_shot_examples]
            examples = [arr for sublist in examples for arr in sublist]
            messages.extend(examples)

        user_prompt = [{"role" : "user", "content" : user_prompt}]
        messages.extend(user_prompt)

        while retries < max_retries:
            try:

                completions = await litellm.acompletion(model = f"openai/{self.model}",
                                                        messages=messages,
                                                        timeout=self.max_timeout_per_request,
                                                        user=f"mr-{os.getenv('ENV')}-concept-extraction",
                                                        temperature=0.0,
                                                        base_url=os.getenv("LITELLM_BASE_URL"),
                                                        api_key=os.getenv("LITELLM_API_KEY")
                                                        )
                return completions
            
            except asyncio.TimeoutError as timeout_err:
                print("\ntimeout err : ",timeout_err)
                print('request sent : ',messages)
                return 'indeterminate'
                    
            except Exception as e:
                print('Exception: {}'.format(e))
                sleep_time = min_sleep_time * (backoff_factor ** retries)
                print(f"Rate limit hit. Retrying in {sleep_time} seconds.")
                await asyncio.sleep(sleep_time) 
                retries += 1
        
        return 'indeterminate'