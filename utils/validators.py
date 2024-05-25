from typing import List, Dict

from utils.custom_llm import CustomLLMBuilder

class StatementValidator:

    def __init__(self,
                 statements:List[str],
                 validation_behavior:str="you are an expert annotator for entity relationships. Given two entities and the relationship between them, return Yes if the relationship is correct, No otherwise",
                ):
        """
        Given a list of 'statements' that can be answered as 'Yes' or 'No', this class will validate the statements using a language model

        args:
            statements : List[str] : a list of statements to be validated. Ideally statements should be in the format '<term1> <relationship> <term2> in the context of ```<context>```'
            validation_behavior : str : the prompt to be used by the language model to validate the statements
        """
        
        self.statements = [str(statement) for statement in statements]
        self.system_prompt = validation_behavior

        self.results = [{"statement": statement,
                         "explanation" : None,
                         "valid": False}
                        for statement in self.statements]

    def validate(self, user_prompt:str="return Yes or No for the following : ") -> List[Dict[str, str]]:
        """
        Validate the statements using a language model. The statements are pre-fixed with the supplied 'user_prompt' before being sent to the language model for validation

        args:
            user_prompt : str : the prompt to be used by the language model to validate the statements. Defaults to "return Yes or No for the following : "

        returns:
            - results : List[Dict[str, str]] : a list of dictionaries containing the validated statements and their corresponding explanations. Each dictionary contains the keys 'statement', 'explanation' and 'valid'
        """

        with CustomLLMBuilder(system_prompt=self.system_prompt) as llm:
            
            batched_prompts = [user_prompt+statement
                            for statement in self.statements]
            
            results = llm.batch_predict(batched_prompts)

        for idx, result in enumerate(results):

            self.results[idx]["explanation"] = result

            if "Yes" in result:
                self.results[idx]["valid"] = True

        return self.results