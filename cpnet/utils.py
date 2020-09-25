import numpy as np
from scipy import sparse
import numba
import networkx as nx
import matplotlib as mpl
import seaborn as sns


def to_adjacency_matrix(net):
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net
        return sparse.csr_matrix(net, dtype=np.float64), np.arange(net.shape[0])
    elif "networkx" in "%s" % type(net):
        return (
            sparse.csr_matrix(nx.adjacency_matrix(net), dtype=np.float64),
            net.nodes(),
        )
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net, dtype=np.float64), np.arange(net.shape[0])


def to_nxgraph(net):
    if sparse.issparse(net):
        return nx.from_scipy_sparse_matrix(net)
    elif "networkx" in "%s" % type(net):
        return net
    elif "numpy.ndarray" == type(net):
        return nx.from_numpy_array(net)


def draw(
    G,
    c,
    x,
    ax,
    font_size=0,
    pos=None,
    cmap=None,
    draw_nodes_kwd={},
    draw_edges_kwd={"edge_color": "#adadad"},
    draw_labels_kwd={},
):
    """
    
    Plot the core-periphery structure in the networks
    
    Params
    ------
    G: networkx.Graph
    c: dict
        - key: node id given by G.noes()
        - value: integer indicating the group id
    x: dict
        - key: node id given by G.noes()
        - value: float indicating the coreness
    ax: matplotlib.axis
    font_size: int
        Font size for node labels. Set 0 to hide the font. 
    pos: dict
        - key: node id given by G.noes()
        - value: tuple (x, y) indicating the location
    cmap: colormap
    draw_nodes_kwd: dict
        Parameter for networkx.draw_networkx_nodes
    draw_edges_kwd: dict
        Parameter for networkx.draw_networkx_edges
    draw_labels_kwd: dict
        Parameter for networkx.draw_networkx_labels
        
    Returns
    ------
    ax: matplotlib.axes
    pos: dict
        - key: node id given by G.noes()
        - value: tuple (x, y) indicating the location
    """

    # Count the number of groups
    num_groups = len(np.unique(np.array(c.values())))

    # Split node into residual and non-residual
    residuals = [d for d in G.nodes() if (c[d] is None) or (x[d] is None)]
    non_residuals = [d for d in G.nodes() if (c[d] is not None) and (x[d] is not None)]

    # Set up the palette
    if cmap is None:
        if num_groups <= 8:
            cmap = sns.color_palette().as_hex()
        elif num_groups <= 20:
            cmap = sns.color_palette("tab20").as_hex()
        else:
            cmap = sns.color_palette("hls", num_groups).as_hex()

    bounds = np.linspace(0, 1, 11)
    norm = mpl.colors.BoundaryNorm(bounds, ncolors=12, extend="both")

    # Calculate the color for each node using the palette
    cmap_coreness = [sns.light_palette(color, n_colors=12).as_hex() for color in cmap]
    node_colors = [
        cmap_coreness[c[d]][norm(x[d]) - 1]
        for i, d in enumerate(G.nodes())
        if x[d] is not None
    ]
    node_edge_colors = [cmap[c[d]] if x[d] == 0 else "#4d4d4d" for d in non_residuals]

    # Set the position of nodes
    if pos is None:
        pos = nx.spring_layout(G)

    # Draw
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, nodelist=non_residuals, ax=ax, **draw_nodes_kwd
    )
    nodes.set_edgecolor(node_edge_colors)
    draw_nodes_kwd["node_size"] = draw_nodes_kwd.get("node_size", 100)
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color="#efefef",
        nodelist=residuals,
        node_shape="s",
        ax=ax,
        **draw_nodes_kwd
    )
    nodes.set_edgecolor("#4d4d4d")

    nx.draw_networkx_edges(G, pos, ax=ax, **draw_edges_kwd)
    if font_size > 0:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, **draw_labels_kwd)
    ax.axis("off")

    return ax, pos
