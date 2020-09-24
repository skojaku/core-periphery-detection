# A Python package for detecting core-periphery structure in networks

This package contains algorithms for detecting core-periphery structure in networks. 
All algorithms are implemented in Python, with speed accelerations by numba, and can be used with a small coding effort.   


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

Retrieve the results

```python
c = algorithm.get_pair_id()
x = algorithm.get_coreness()
```

`c` and `x` are python dict objects that take node labels (i.e., `G.nodes()`) as keys. 
- The values of `c` are integers indicating group ids: nodes having the same integer belong to the same group. 
- The values of `x` are float values indicating coreness ranging between 0 and 1. A larger value indicates that the node is closer to the core. In case of discrete core-periphery structure, the corenss can only take 0 or 1, with x[i]=1 or =0 indicating that node i belongs to a core or a periphery, respectively.

For example,
 
```python
c = {A: 0, B: 1, C: 0, D: 2 ...} 
x = {A: 1, B: 1, C: 0, D: 1 ...}
```

means that nodes A and C belong to group 1, A is a core node, and C is a peripheral node of the group A belongs to.


All algorithms implemented in this package have the same inferface. This means that you can use other algorithms by changing `conet.KM_config` to, for example, `cpnet.BE`. See the list of algorithms as follows:

| Algorithm | Reference |
|-----------|-----------|
| [cpnet.BE](cpnet/BE.py) | [S. P. Borgatti and M. G. Everett. Models of core/periphery structures. Social Networks, 21, 375–395, 2000](https://www.sciencedirect.com/science/article/abs/pii/S0378873399000192)|
| [cpnet.MINRES](cpnet/MINRES.py)  | [S. Z. W. Lip. A fast algorithm for the discrete core/periphery bipartitioning problem. Preprint arXiv: 1102.5511, 2011](https://arxiv.org/abs/1102.5511) |
| [cpnet.LowRankCore](cpnet/Cucuringu.py) <br> [cpnet.LapCore](cpnet/Cucuringu.py) <br> [cpnet.LapSgnCore](cpnet/Cucuringu.py) | [M. Cucuringu, P. Rombach, S. H. Lee, and M. A. Porter. Detection of core-periphery structure in networks using spectral methods and geodesic paths. European Journal of Applied Mathematics, 846–887, 2016](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/detection-of-coreperiphery-structure-in-networks-using-spectral-methods-and-geodesic-paths/A08BE0DA1A8AD7C58C24AF53AA134729)|
| [cpnet.Rombach](cpnet/Rombach.py)  | [P. Rombach, M. A. Porter, J. H. Fowler, and P. J. Mucha. Core-Periphery Structure in Networks (Revisited). SIAM Review, 59, 619–646, 2017](https://epubs.siam.org/doi/10.1137/17M1130046) |
| [cpnet.Rossa](cpnet/Rossa.py)  | [F. Rossa, F. Dercole, and C. Piccardi. Profiling core-periphery network structure by random walkers. Scientific Reports, 3, 1467, 2013](https://www.nature.com/articles/srep01467) |
| [cpnet.Surprise](cpnet/Surprise.py) | [J. van Lidth de Jeude, G. Caldarelli, T. Squartini. Detecting Core-Periphery Structures by Surprise. EPL, 125, 2019](https://epljournal.edpsciences.org/articles/epl/abs/2019/06/epl19592/epl19592.html) |
| [cpnet.KM_ER](cpnet/KM_ER.py) | [S. Kojaku and N. Masuda. Finding multiple core-periphery pairs in networks. Physical Review E, 96, 052313, 2017](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.052313) |
| [cpnet.KM_config](cpnet/KM_config.py) <br> [cpnet.Divisive](cpnet/Divisive.py) | [S. Kojaku and N. Masuda. Core-periphery structure requires something else in networks. New Journal of Physics, 20, 043012, 2018](https://iopscience.iop.org/article/10.1088/1367-2630/aab547)|

Some algorithms have tuning parameters. Please see the source code for the parameters specific to each algorithm. 

One can detect

- a single pair of a core and a periphery using
  - `cpnet.BE`, `cpnet.MINRES`, `cpnet.LapCore`, `cpnet.LapSgnCore`, `cpnet.Surprise`
- multiple pairs of a core and a periphery using
  - `cpnet.KM_ER`, `cpnet.KM_config`, `cpnet.Divisive`
- a continuous spectrum between a core and a periphery using
  - `cpnet.Rombach`, `cpnet.Rossa` 

## Statistical test

The core-periphery structure detected by any algorithm may be dubious. For example, an algorithm labels nodes as core or periphery even if the network is a regular random graph, which is supposed not to have core-periphery structure. 

It is crucial to inspect whether the detected core-periphery structure is significant, which is why the statistical test comes in. This package has an implementation of a statistical test, *q-s test*, in which one runs the algorithm used to detect the core-periphery structure in question to many randomized networks. If randomized networks do not yield a core-periphery structure stronger than that detected for the input network, the core-periphery structure detected for the input network is considered as significant. See papers [here](https://www.nature.com/articles/s41598-018-25560-z) and [here](https://iopscience.iop.org/article/10.1088/1367-2630/aab547) for the method.

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
- `significant` is a boolean list, where `significant[c]=True` or `significant[c]=False` indicates that the cth core-periphery pair is significant or insignificant, respectively. 
- `p_values` is a float list, where `p_values[c]` is the p-value for the *c*[でいいのかな。c を italic にする。または c-th にする。cth はNGなので。他の cth も同様に直す。]th core-periphery pair under a null model (default is the configuration model).

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


# Examples
増田さんへ
ここに2つか3つくらい例を載せようと思います。最初に使い方の手順を簡単に書きましたが、実際にどう使うのかを説明するために実践例が必要かなと思い、この提案をしています。
例はpolitical blog, worldwide airport networkにアルゴリズムを適用して可視化するところまでをjupyter notebookに書き、リンクをここに貼ろうと考えています。github上はjupyter notebookを図入りで表示してくれるので、興味も引きやすいかなと思います。 [絶対ある方がよい。例は多くてもよい。]
- [Example 1 (Detection of core-periphery structure)](https://github.com/skojaku/core-periphery-detection/blob/add-notebook/notebooks/exampl1.ipynb)
- [Example 2 (Statistical test)](https://github.com/skojaku/core-periphery-detection/blob/add-notebook/notebooks/exampl2.ipynb)
