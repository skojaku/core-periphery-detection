# Core-periphery detection algorithm 



# Installation


For pip, 

.. code-block:: bash

  $ pip install cpnet

For conda,   

.. code-block:: bash

  $ conda install -c conda-forge -c skojaku cpnet 


Dependency:
- Python version >=3.6
- decorator==4.4.2
- joblib==0.16.0
- llvmlite==0.33.0
- networkx==2.5rc1
- numba==0.50.0
- numpy==1.19.1
- scipy==1.5.2
- simanneal==0.5.0

## Usage 

Load an algorithm for detecting core-periphery structure in networks:

```python
import cpnet 
algorithm = cpnet.KM_config()
```

Pass a graph object (networkx.Graph) to the algorithm:

```python
import networkx as nx
G = nx.karate_club_graph()
algorithm.detect(G)
```

Retrieve the results

```python
c = algorithm.get_pair_id()
x = algorithm.get_coreness()
```

:python:`c` and :python:`x` are python dict objects that takes node labels (i.e., :python:`G.nodes()`) as keys. 
The values of :python:`c` are integers indicating group ids: nodes having the same integer belong to the same group. 
The values of :python:`x` are float values indicating coreness, i.e., a level of belongingness to the core.
For example,
 
.. code-block:: python

   c = {A: 0, B: 1, C: 0, D: 2 ..., 
   x = {A: 1, B: 1, C: 0, D: 1 ...,

mean nodes A, C belong to group 1, A is a core, and C is a periphery for the group. 
