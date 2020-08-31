# A Python package for detecting core-periphery structure in networks

This package contains several algorithms for detecting core-periphery structure in networks. 
All algorithms are implemented in python, with a speed enhancement by numba, and can be used with minimal coding effort.   


# Installation

Before installing this package, make sure that you have a **Python with version 3.6 or above**.

There are two ways to install this package, *conda* or *pip*. conda is recommended if you have a conda environment. Otherwise, use pip for installation.  

For conda,   

```bash
conda install -c conda-forge -c skojaku cpnet 
```

For pip, 

```bash
pip install cpnet
```

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

# Usage

This package consists of two submodules:
- Set of algorithms for detecting core-periphery structure in networks
- A statistical test for core-periphery structure

## Core-periphery detection

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

Retrieve the results

```python
c = algorithm.get_pair_id()
x = algorithm.get_coreness()
```

`c` and `x` are python dict objects that take node labels (i.e., `G.nodes()`) as keys. 
The values of `c` are integers indicating group ids: nodes having the same integer belong to the same group. 
The values of `x` are float values indicating coreness, i.e., a level of belongingness to the core.
For example,
 
```python
c = {A: 0, B: 1, C: 0, D: 2 ..., 
x = {A: 1, B: 1, C: 0, D: 1 ...,
```

mean nodes A, C belong to group 1, A is a core, and C is a periphery for the group.


All algorithms implemented in this package have the same inferface. This means that you can use other algorithms by changing `conet.KM_config` to, for example, `cpnet.BE`. See the list of algorithms as follows:

| Algorithm | Reference |
|-----------|-----------|
| [cpnet.BE](cpnet/BE.py) | S. P. Borgatti and M. G. Everett. Models of core/periphery structures. Soc. Netw., 21, 375–395, 2000 |
| [cpnet.MINRES](cpnet/MINRES.py)  | S. Z. W. Lip. A fast algorithm for the discrete core/periphery bipartitioning problem. arXiv, 2011 |
| [cpnet.LowRankCore](cpnet/Cucuringu.py)  | M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter Detection of core-periphery structure in networks using spectral methods and geodesic paths. Euro. J. Appl. Math., 846–887, 2016 |
| [cpnet.LapCore](cpnet/Cucuringu.py)  | M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter Detection of core-periphery structure in networks using spectral methods and geodesic paths. Euro. J. Appl. Math., 846–887, 2016 |
| [cpnet.LapSgnCore](cpnet/Cucuringu.py) | M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter Detection of core-periphery structure in networks using spectral methods and geodesic paths. Euro. J. Appl. Math., 846–887, 2016 |
| [cpnet.Rombach](cpnet/Rombach.py)  | P. Rombach, M. A. Porter, J. H. Fowler, and P. J. Mucha. Core-Periphery Structure in Networks (Revisited). SIAM Review, 59, 619–646, 2017 |
| [cpnet.Rossa](cpnet/Rossa.py)  | F. Rossa, F. Dercole, and C. Piccardi. Profiling core-periphery network structure by random walkers. Scientific Reports, 3, 1467, 2013 |
| [cpnet.Surprise](cpnet/Surprise.py) | J. van Lidth de Jeude, G. Caldarelli, T. Squartini. Detecting Core-Periphery Structures by Surprise. EPL, 125, 2019 |
| [cpnet.KM_ER](cpnet/KM_ER.py) | S. Kojaku and N. Masuda. Finding multiple core-periphery pairs in networks. Phys. Rev. 96, 052313, 2017 |
| [cpnet.KM_config](cpnet/KM_config.py) | S. Kojaku and N. Masuda. Core-periphery structure requires something else in networks. New J. Phys., 20, 043012, 2018 |
| [cpnet.Divisive](cpnet/Divisive.py) | S. Kojaku and N. Masuda. Core-periphery structure requires something else in networks. New J. Phys., 20, 043012, 2018 |

Some algorithms have tuning parameters. Please see the source code for the parameters specific to the algorithms. 

### Detectable core-periphery structure 

- Single pair of a core and a periphery:
  - `cpnet.BE`, `cpnet.MINRES`, `cpnet.LapCore`, `cpnet.LapSgnCore`, `cpnet.Surprise`
- Multiple pairs of a core and a periphery 
  - `cpnet.KM_ER`, `cpnet.KM_config`, `cpnet.Divisive`
- Continuous spectrum between a core and a periphery:  
  - `cpnet.Rombach`, `cpnet.Rossa` 

## Statistical test


The statistical test can be performed by 

```python
sig_c, sig_x, significant, p_values = cpnet.qstest(c, x, G, algorithm, significance_level = 0.05, num_of_thread = 4)
```
- `c` and `x` are the core-periphery pairs in question that will be tested by the statistical test
- `G` is the graph object (Networkx.Graph)
- `algorithm` is the algorithm that you used to get `c` and `x`
- `significance_level` is the significance level. (Optional; Default is 0.05)
- `num_of_thread` Number of threads to perform the statistical test (Optional; Default is 4)
- `sig_c` and `sig_x` are dict objects taking node names as its keys. The values of the dict objects are the same as the `c` and `x` but `None` for the nodes belonging to the insignificant core-periphery pairs. 
- `significant` is a boolean list, where `significant[c]=True` or `significant[c]=False` indicates that the cth core-periphery pair is significant or insignificant, respectively. 
- `p_values` is a float list, where `p_values[c]` is the p-value for the cth core-periphery pair under a null model (default is the configuration model).

Some core-periphery pairs may be deemed as insignificant but have a p-value smaller than the prescribed significance level. This is because the judgement of statistical significance is made based on a different significance level; the qs-test automatically adjusts the significance level using the Sidak correction in order to control for the false positives due to the multiple comparison problem.    


### Use a different null model 

The p-value is computed using the configuration model as the null model. You may use a different null model by passing a user-defined function as the `null_model` argument to `qstest`. 
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
