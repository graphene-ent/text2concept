from typing import List, Dict

class DataChunk:
    def __init__(self,
                 id: int,
                 text: str,
                 size: int):
        """
        A class to represent a DataChunk.

        args:
            id: UUID - unique identifier for the chunk
            text: str - the text content of the chunk
            size: int - number of words in the chunk
        
        returns:
            None
        """
        self.id = id
        self.text = text
        self.size = size

class BaseTextChunker:

    def __init__(self,
                 docs:List[str]):
        """
        Simple base class to chunk text data into smaller pieces.

        args:
            docs:List[str] - list of strings to be chunked
        
        returns:
            None
        """
        
        self.docs = [str(doc) for doc in docs]
        self.docs = [doc for doc in self.docs
                    if doc.strip() not in [None, 'None', "", 'nan', 'NaN']]

        self.chunks = []

    def chunk_text(self,
                   chunk_size:int=200) -> List[DataChunk]:
        """
        Abstract method to chunk text data into smaller pieces.

        args:
            chunk_size:int - maximum number of words in each chunk. Defaults to 200.

        returns:
            List[DataChunk] - list of DataChunk objects. Each DataChunk object has the following attributes:
                id:(UUID) - unique id for the chunk
                text:(str) - text of the chunk
                size:(int) - number of words in the chunk
        """

        return NotImplementedError

    def get_output(self,
                   chunk_size:int=200) -> List[Dict[str, str]]:
        """
        calls chunk_text() method to chunk text data, returns the chunks formed

        args:
            chunk_size:int - maximum number of words in each chunk. Defaults to 200.
        """

        self.chunks = self.chunk_text(chunk_size=chunk_size)
        
        return self.chunks