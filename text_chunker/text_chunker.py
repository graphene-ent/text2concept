from typing import List
from uuid import uuid4

from text_chunker.base_chunker import BaseTextChunker, DataChunk

from utils.constants import PUNCTUATIONS

class SimpleChunker(BaseTextChunker):

    def __init__(self,
                 docs:List[str]):

        super().__init__(docs=docs)

    def chunk_text(self,
                    chunk_size:int=200) -> List[DataChunk]:
        """
        Method to chunk text data into smaller pieces. The chunks are formed based on the chunk_size parameter as a 'guideline' to prioritize keeping the entire document in the same chunk.
        If the chunk_size is exceeded, the current document is NOT split into two chunks. Instead, the current document is added to the current chunk.

        args:
            chunk_size:int - number of words in each chunk. Defaults to 200.

        returns:
            List[DataChunk] - list of DataChunk objects. Each DataChunk object has the following attributes:
                id:(UUID) - unique id for the chunk
                text:(str) - text of the chunk
                size:(int) - number of words in the chunk
        """

        chunks = []
        current_chunk_size, current_chunk = 0, ""

        for doc in self.docs:

            if current_chunk_size >= chunk_size:

                chunks.append(DataChunk(id=uuid4(),
                                        text=current_chunk,
                                        size=len(current_chunk.split(" "))
                                        )
                            )
                
                #reset the current_chunk and current_chunk_size
                current_chunk = ""
                current_chunk_size = 0

            else:

                #add a period at the end of the document if it doesn't end with a punctuation
                if doc.strip()[-1] not in PUNCTUATIONS:
                    doc+". "

                current_chunk += doc
                current_chunk_size += len(doc.strip().split(" "))

        if current_chunk:
            chunks.append(DataChunk(id=uuid4(),
                                    text=current_chunk,
                                    size=len(current_chunk.split(" "))
                                    )
                        )
        return chunks