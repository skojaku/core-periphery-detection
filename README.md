# Core-periphery detection algorithm 



# Installation


For pip, 

```bash
  $ pip install cpnet
```

For conda,   

```bash
  $ conda install -c conda-forge -c skojaku cpnet 
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

`c` and `x` are python dict objects that takes node labels (i.e., `G.nodes()`) as keys. 
The values of `c` are integers indicating group ids: nodes having the same integer belong to the same group. 
The values of `x` are float values indicating coreness, i.e., a level of belongingness to the core.
For example,
 
```python
   c = {A: 0, B: 1, C: 0, D: 2 ..., 
   x = {A: 1, B: 1, C: 0, D: 1 ...,
```

mean nodes A, C belong to group 1, A is a core, and C is a periphery for the group.

List of algorithm:

| Algorithm | Reference |
|-----------|-----------|
| cpnet.BE  | S. P. Borgatti and M. G. Everett. Models of core/periphery structures. Soc.~Netw., 21(4):375–395, 2000 |
| cpnet.MINRES  | S. Z. W.~ Lip. A fast algorithm for the discrete core/periphery bipartitioning problem. arXiv, pages 1102.5511, 2011 |
| cpnet.LowRankCore  | M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter Detection of core-periphery structure in networks using spectral methods and geodesic paths. Euro. J. Appl. Math., 27:846–887, 2016. |
| cpnet.LapCore  | M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter Detection of core-periphery structure in networks using spectral methods and geodesic paths. Euro. J. Appl. Math., 27:846–887, 2016. |
| cpnet.LapSgnCore  | M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter Detection of core-periphery structure in networks using spectral methods and geodesic paths. Euro. J. Appl. Math., 27:846–887, 2016. |
| cpnet.Rombach  | P. Rombach, M. A. Porter, J. H. Fowler, and P. J. Mucha. Core-Periphery Structure in Networks (Revisited). SIAM Review, 59(3):619–646, 2017 |
| cpnet.Rossa  | F. Rossa, F. Dercole, and C. Piccardi. Profiling core-periphery network structure by random walkers. Scientific Reports, 3, 1467, 2013 |
| cpnet.Surprise | J. van Lidth de Jeude, G. Caldarelli, T. Squartini. Detecting Core-Periphery Structures by Surprise. EPL, 125, 2019 |
| cpnet.KM_ER | S. Kojaku and N. Masuda. Finding multiple core-periphery pairs in network. Phys. Rev. 96, 052313, 2017 |
| cpnet.KM_config | S. Kojaku and N. Masuda. Core-periphery structure requires something else in networks. New J. Phys. 2018 |
| cpnet.Divisive | S. Kojaku and N. Masuda. Core-periphery structure requires something else in networks. New J. Phys. 2018 |

## Statistical test


The statistical test can be performed by 

```python
   sig_c, sig_x, significant, p_values = cp.qstest(c, x, G, algorithm)
```
 - `sig_c` and `sig_x` are dict objects taking node name as its keys. The values of the dict objects are the same as the `c` and `x` but `None` for the nodes belonging to the insignificant core-periphery pairs. 
 - `significant` is a boolean list, where `significant[c]=True` or `significant[c]=False` indicates that the cth core-periphery pair is significant or insignificant, respectively. 
 - `p_values` is a float list, where `p_values[c]` is the p-value for the cth core-periphery pair under a null model (default is the configuration model).

The statistical test is performed at the individual groups. Thus, caution should be taken to a problem known as multiple testing problems, i.e., even if all tests should be insignificant,  as the number of tests increases, the chance of getting a significantly small p-value in one of the tests increases (false positives). 
To circumvent this problem, `qstest` lowers the significance level using the Sidak correction and thus controls the chance of false positives. 
The `significant` is the result after applying the Sidak correction. 


