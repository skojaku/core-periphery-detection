# A Python package for detecting core-periphery structure in networks

This package contains algorithms for detecting core-periphery structure in networks. 
All algorithms are implemented in Python, with speed accelerations by numba, and can be used with a small coding effort.   

### APIs

See [documentation]()


# Installation

Before installing this package, make sure that you have a **Python with version 3.6 or above**.

There are two ways to install this package, *conda* or *pip*. conda is recommended if you have a conda environment. Otherwise, use pip.  

For conda,   

```bash
conda install -c conda-forge -c skojaku cpnet 
```

For pip, 

```bash
pip install cpnet
```

### Dependency:

See requirements.txt

# Usage

This package consists of the following two submodules:
- A set of algorithms for detecting core-periphery structure in networks
- Codes for a statistical test for core-periphery structure

## Detection of core-periphery structure

Load an algorithm for detecting core-periphery structure in networks:

```python
import cpnet 
algorithm = cpnet.KM_config()
```

Pass a graph object (networkx.Graph or adjacency matrix in scipy.sparse format) to the algorithm:

```python
import networkx as nx
G = nx.karate_club_graph()
algorithm.detect(G)
```

Retrieve the results:

```python
c = algorithm.get_pair_id()
x = algorithm.get_coreness()
```

`c` and `x` are python dict objects that take node labels (i.e., `G.nodes()`) as keys. 
- The values of `c` are integers indicating group ids: nodes having the same integer belong to the same group. 
- The values of `x` are float values indicating coreness ranging between 0 and 1. A larger value indicates that the node is closer to the core. In case of discrete core-periphery structure, the corenss can only take 0 or 1, with x[i]=1 or =0 indicating that node i belongs to a core or a periphery, respectively.

For example,
 
```python
c = {A: 0, B: 1, C: 0, ...} 
x = {A: 1, B: 1, C: 0, ...}
```

means that nodes A and C belong to group 0, and B belongs to a different group 1. Furtheremore, A and B are core nodes, and C is a peripheral node.

All algorithms implemented in this package have the same inferface. This means that you can use other algorithms by changing `cpnet.KM_config` to, for example, `cpnet.BE`. See the list of algorithms as follows:

| Algorithm | Reference |
|-----------|-----------|
| [cpnet.BE](cpnet/BE.py) | [S. P. Borgatti and M. G. Everett. Models of core/periphery structures. Social Networks, 21, 375–395, 2000](https://www.sciencedirect.com/science/article/abs/pii/S0378873399000192)|
| [cpnet.Lip](cpnet/Lip.py)  | [S. Z. W. Lip. A fast algorithm for the discrete core/periphery bipartitioning problem. Preprint arXiv: 1102.5511, 2011](https://arxiv.org/abs/1102.5511) |
| [cpnet.LowRankCore](cpnet/Cucuringu.py) <br> [cpnet.LapCore](cpnet/Cucuringu.py) <br> [cpnet.LapSgnCore](cpnet/Cucuringu.py) | [M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter. Detection of core-periphery structure in networks using spectral methods and geodesic paths. European Journal of Applied Mathematics, 846–887, 2016](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/detection-of-coreperiphery-structure-in-networks-using-spectral-methods-and-geodesic-paths/A08BE0DA1A8AD7C58C24AF53AA134729)|
| [cpnet.Rombach](cpnet/Rombach.py)  | [P. Rombach, M. A. Porter, J. H. Fowler, and P. J. Mucha. Core-Periphery Structure in Networks (Revisited). SIAM Review, 59, 619–646, 2017](https://epubs.siam.org/doi/10.1137/17M1130046) |
| [cpnet.Rossa](cpnet/Rossa.py)  | [F. Rossa, F. Dercole, and C. Piccardi. Profiling core-periphery network structure by random walkers. Scientific Reports, 3, 1467, 2013](https://www.nature.com/articles/srep01467) |
| [cpnet.Surprise](cpnet/Surprise.py) | [J. van Lidth de Jeude, G. Caldarelli, T. Squartini. Detecting Core-Periphery Structures by Surprise. EPL, 125, 2019](https://epljournal.edpsciences.org/articles/epl/abs/2019/06/epl19592/epl19592.html) |
| [cpnet.KM_ER](cpnet/KM_ER.py) | [S. Kojaku and N. Masuda. Finding multiple core-periphery pairs in networks. Physical Review E, 96, 052313, 2017](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.052313) |
| [cpnet.KM_config](cpnet/KM_config.py) <br> [cpnet.Divisive](cpnet/Divisive.py) | [S. Kojaku and N. Masuda. Core-periphery structure requires something else in networks. New Journal of Physics, 20, 043012, 2018](https://iopscience.iop.org/article/10.1088/1367-2630/aab547)|

Some algorithms have tuning parameters. Please see the source code for the parameters specific to each algorithm. 

One can detect

- a single pair of a core and a periphery using
  - `cpnet.BE`, `cpnet.Lip`, `cpnet.LapCore`, `cpnet.LapSgnCore`, `cpnet.Surprise`
- multiple pairs of a core and a periphery using
  - `cpnet.KM_ER`, `cpnet.KM_config`, `cpnet.Divisive`
- a continuous spectrum between a core and a periphery using
  - `cpnet.Rombach`, `cpnet.Rossa` 

## Statistical test

The core-periphery structure detected by any algorithm can be systematic artifacts; even for networks without core-periphery structure such as regular graphs and random graphs, an algorithm labels nodes as core or periphery. 

To filter out spurious core-periphery structure, this package provides an implementation of a statistical test, *q-s test*.
In this test, one generate many randomized networks and detect core-periphery structure with the algorithm used to detect the core-periphery structure in question.  The core-periphery structure detected for the input network is considered as significant if it is stronger than those detected in randomized networks. See papers [here](https://www.nature.com/articles/s41598-018-25560-z) and [here](https://iopscience.iop.org/article/10.1088/1367-2630/aab547) for the method.

To carry out the statistical test, run 

```python
sig_c, sig_x, significant, p_values = cpnet.qstest(c, x, G, algorithm, significance_level = 0.05, num_of_thread = 4)
```
- `c` and `x` are the core-periphery pairs in question that will be tested by the statistical test
- `G` is the graph object (Networkx.Graph)
- `algorithm` is the algorithm that you used to get `c` and `x`
- `significance_level` is the significance level. (Optional; default is 0.05)
- `num_of_thread` Number of threads to perform the statistical test (Optional; default is 4)
- `sig_c` and `sig_x` are dict objects taking node names as its keys. The values of the dict objects are the same as the `c` and `x` but `None` for the nodes belonging to the insignificant core-periphery pairs. 
- `significant` is a boolean list, where `significant[k]=True` or `significant[k]=False` indicates that the k-th core-periphery pair is significant or insignificant, respectively. 
- `p_values` is a float list, where `p_values[k]` is the p-value for the k-th core-periphery pair under a null model (default is the configuration model).

Some core-periphery pairs have a p-value smaller than the prescribed significance level but deemed as insignificant. This is because the statistical significance is adjusted to control for the false positives due to the multiple comparison problem.    


### Use a different null model 

The p-value is computed using the configuration model as the null model. One can use a different null model by passing a user-defined function as the `null_model` argument to `qstest`. 
For example, to use the Erdős–Rényi random graph as the null model, define  

```python
def erdos_renyi(G):
    n = G.number_of_nodes()
    p = nx.density(G)
    return nx.fast_gnp_random_graph(n, p)
```

Then, pass it to the argument of the qstest:

```python
sig_c, sig_x, significant, p_values = cpnet.qstest(
    c, x, G, algorithm, significance_level=0.05, null_model=erdos_renyi
)
```
## Visualization

`cpnet` implements a drawing function based on networkx. 

```python
ax, pos = cpnet.draw(G, c, x, ax, draw_edge=False, draw_nodes_kwd={}, draw_edges_kwd={}, draw_labels_kwd={})
```
- `G` is the graph object (Networkx.Graph)
- `c` and `x` are the core-periphery pairs
- `ax` is the matplotlib axis
- `draw_edge` is a boolean (Optional; Default False). Set `draw_edge = True` not to draw the edges (recommended if the network is large)
- `draw_nodes_kwd={}`, `draw_edges_kwd={}`, and `draw_labels_kwd={}` are the keywords that are passed to networkx.draw_network_nodes, networkx.draw_network_edges, and networkx.draw_network_labels, respectively (see the [networkx documentation](https://networkx.github.io/documentation/stable/reference/drawing.html)). Useful when refining the figure.
- `pos` is a dict object. The keys are the node ids given by G.nodes(). The values are tuples (x, y) indicating the positions of nodes.
- See the [code](cpnet/utils.py) for other parameters.

`cpnet` also implements a function for drawing an interactive figure based on plotly.

```python
fig = cpnet.draw_interactive(G, c, x, hover_text)
```
- `G` is the graph object (Networkx.Graph)
- `c` and `x` are the core-periphery pairs
- `hover_text` is a dict object (optional), where the key is the node id given by G.nodes() and the value is the text to show in the toolbox.



The drawing functions are demonstrated in the example notebook. See 

# Examples
- [Example 1 (Detection of core-periphery structure)](notebooks/example1.ipynb)
- [Example 2 (Statistical test)](notebooks/example2.ipynb)
- [Example 3 (Case study: Pilitical blog network)](notebooks/example3.ipynb)
- [Example 4 (Case study: Airport network)](notebooks/example4.ipynb)
