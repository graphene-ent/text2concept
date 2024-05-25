from typing import List, Dict

import networkx as nx
import pandas as pd

from pyvis.network import Network

def visualize_as_graph(concepts:List[Dict[str,str]],
                       output_filename="graph.html") -> None:
    """
    Visualize the extracted concepts as a graph using the pyvis library

    args:
        - concepts : List[Dict[str,str]] : list of dictionaries containing the extracted concepts. Each dictionary should have the keys 'node_1', 'node_2', and 'relation'
        - output_filename : str : the name of the output file to save the graph to. Defaults to 'graph.html'

    returns:
        - None
    """

    df = pd.DataFrame(concepts)
    G = nx.from_pandas_edgelist(df, 'node_1', 'node_2', edge_attr='relation')

    net = Network(notebook=True)
    net.from_nx(G)

    node_freq = pd.concat([df['node_1'], df['node_2']]).value_counts().to_dict()

    for node in net.nodes:
        node["color"] = "purple"
        node["size"] = node_freq.get(node["id"], 1)  # Set size based on frequency
        node["label"] = "node : "+node["id"]

    for edge in net.edges:
        edge["color"] = "green"
        edge["label"] = "relation : " +str(G[edge["from"]][edge["to"]]['relation'])

    if output_filename.endswith(".html"):
        net.show(output_filename)
    
    else:
        net.show("graph.html")