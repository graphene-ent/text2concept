# text2concept

Convert any document into a graph of contextually relevant concepts


## Setup

* create a virtual environment using virtualenv or pyenv (python3.10 recommended)
* activate environment, install the requirements file

## Starter Script for Running

```python
import pandas as pd

from text_chunker.text_chunker import SimpleChunker
from concept_extractor.concept_extractor import ConceptExtractor

from utils.helpers import visualize_as_graph

#load some data
input_data = pd.read_csv("./sample data.csv")

chunker = SimpleChunker(docs=input_data["text"])
chunks = chunker.get_output()

#extract concepts from data
concept_extractor = ConceptExtractor(chunks=chunks)
concepts = concept_extractor.get_output()

#visualize the concepts as a graph!
visualize_as_graph(concepts=concepts, output_filename="concept_graph.html")

```
