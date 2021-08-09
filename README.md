# Graphsite
Graphsite is a software written in Python that reads a pocket (in .mol2) and compute its graph representation. In each generated graph, each atom represents a node. If the distance between two atoms are less than or equal to a threshold (default is 4.5 Angstrom), an undirected edge is formed between these two atoms. The edge attribute is the number of chemical bond(s) on this edge.

## Intallation
```
pip install graphsite
```

## Usage
Graphsite is invented for applications which are based on graph neural networks (GNNs). However, the scope of this tool is not limited to deep graph learning. It can be used in any application that requires graph representaions of proteins/binding sites.  

### Basic usage
The main module of Graphsite is a callable class which reads 3 files as input:
```python
from graphsite import Graphsite

graphsite = Graphsite()
node_feature, edge_index, edge_attr = pocket_to_graph(
        # path to the .mol2 file of pocket
        mol_path=mol_path, 

        # path to the .profile file of pocket which
        # contains the sequence entropy node feature
        profile_path=profile_path, 

        # path to the .popsa file of pocket which contains
        # the solvent-accessible surface area node feature
        pop_path=pop_path
    )
```
The ```node_feature```, ```edge_index```, and ```edge_attr``` are numpy arrays. For more information about input paramters and output formats, see help:
```
```

### A Pytorch example
Below is an example where the output matrices of graphsite are used to create graphs for [Pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/).
```python
from torch_geometric.data import Data

data = Data(
    x=node_feature,
    edge_index=edge_index,
    edge_attr=edge_attr)
)
```
For a complete deep learning example, please see [Graphsite-classifier](https://github.com/shiwentao00/Graphsite-classifier), where we build a graph classifier over 14 classes of binding pockets.

## Cite our work
Please cite our paper if you find this package useful in your project :)
```
Currently under peer review
```

## Feedback
If you have any questions or suggestions, please submit an issue or pull request. Anyone is welcome to contribute :)
