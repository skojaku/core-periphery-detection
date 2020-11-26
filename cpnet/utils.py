import numpy as np
from scipy import sparse
import numba
import networkx as nx
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go


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


def set_node_colors(G, c, x, cmap, max_num=None):
    # Count the number of groups
    cvals = np.array(list(c.values()))
    cids = np.array(list(set(cvals).difference(set([None]))))
    num_groups = len(cids) 

    if max_num is None:
        max_num = num_groups

    # Set up the palette
    if cmap is None:
        if np.minimum(max_num, num_groups) <= 8:
            cmap = sns.color_palette().as_hex()
        elif np.minimum(max_num, num_groups) <= 20:
            cmap = sns.color_palette("tab20").as_hex()
        else:
            cmap = sns.color_palette("hls", num_groups).as_hex()

    # Calc size of groups
    freq = np.array([np.sum(cid == cvals) for cid in cids])
    cmap = dict(zip(cids[np.argsort(-freq)], [cmap[i] for i in range(max_num)]))

    bounds = np.linspace(0, 1, 11)
    norm = mpl.colors.BoundaryNorm(bounds, ncolors=12, extend="both")
    
    # Calculate the color for each node using the palette
    cmap_coreness = {k:sns.light_palette(v, n_colors=12).as_hex() for k, v in cmap.items()}
    cmap_coreness_dark = {
        k:sns.dark_palette(v, n_colors=12).as_hex() for k, v in cmap.items()
    }
    node_colors = {
        d:(cmap_coreness[c[d]][norm(x[d]) - 1] if (x[d] is not None and c[d] in cmap_coreness) else "#4d4d4d")
        for i, d in enumerate(G.nodes())
    }
    node_edge_colors = {}
    for i, d in enumerate(G.nodes()):
        if (x[d] is None) or (c[d] not in cmap_coreness):
            node_edge_colors[d]= "#4d4d4d"
        else:
            node_edge_colors[d]= cmap_coreness_dark[c[d]][-norm(x[d])]
    return node_colors, node_edge_colors


def draw(
    G,
    c,
    x,
    ax,
    draw_edge=True,
    font_size=0,
    pos=None,
    cmap=None,
    max_colored_group_num=None,
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
        - key: node id given by G.nodes()
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

    node_colors, node_edge_colors = set_node_colors(
        G, c, x, cmap, max_colored_group_num
    )

    # Set the position of nodes
    if pos is None:
        pos = nx.spring_layout(G)

    # Split node into residual and non-residual
    residuals = [d for d in G.nodes() if (c[d] is None) or (x[d] is None)]
    non_residuals = [d for d in G.nodes() if (c[d] is not None) and (x[d] is not None)]
    
    # Draw
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[
            node_colors[d] for i, d in enumerate(G.nodes()) if x[d] is not None
        ],
        nodelist=non_residuals,
        ax=ax,
        **draw_nodes_kwd
    )
    if nodes is not None:
        nodes.set_edgecolor([node_edge_colors[r] for r in non_residuals])

    draw_nodes_kwd_residual = draw_nodes_kwd.copy()
    draw_nodes_kwd_residual["node_size"] = 0.1 * draw_nodes_kwd.get("node_size", 100)
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color="#efefef",
        nodelist=residuals,
        node_shape="s",
        ax=ax,
        **draw_nodes_kwd_residual
    )
    if nodes is not None: 
        nodes.set_edgecolor("#4d4d4d")

    if draw_edge:
        nx.draw_networkx_edges(G, pos, ax=ax, **draw_edges_kwd)

    if font_size > 0:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, **draw_labels_kwd)

    ax.axis("off")

    return ax, pos


def draw_interactive(G, c, x, hover_text=None, node_size=10.0, pos=None, cmap=None):

    node_colors, node_edge_colors = set_node_colors(G, c, x, cmap)

    if pos is None:
        pos = nx.spring_layout(G)

    nodelist = [d for d in G.nodes()]
    group_ids = [c[d] if c[d] is not None else "residual" for d in nodelist]
    coreness = [x[d] if x[d] is not None else "residual" for d in nodelist]
    node_size_list = [(x[d] + 1) if x[d] is not None else 1 / 2 for d in nodelist]

    pos_x = [pos[d][0] for d in nodelist]
    pos_y = [pos[d][1] for d in nodelist]
    df = pd.DataFrame(
        {
            "x": pos_x,
            "y": pos_y,
            "name": nodelist,
            "group_id": group_ids,
            "coreness": coreness,
            "node_size": node_size_list,
        }
    )
    df["marker"] = df["group_id"].apply(
        lambda s: "circle" if s != "residual" else "square"
    )

    df["hovertext"] = df.apply(
        lambda s: "{ht}<br>Group: {group}<br>Coreness: {coreness}".format(
            ht="Node %s" % s["name"]
            if hover_text is None
            else hover_text.get(s["name"], ""),
            group=s["group_id"],
            coreness=s["coreness"],
        ),
        axis=1,
    )

    fig = go.Figure(
        data=go.Scatter(
            x=df["x"],
            y=df["y"],
            marker_size=df["node_size"],
            marker_symbol=df["marker"],
            hovertext=df["hovertext"],
            hoverlabel=dict(namelength=0),
            hovertemplate="%{hovertext}",
            marker={
                "color": node_colors,
                "sizeref": 1.0 / node_size,
                "line": {"color": node_edge_colors, "width": 1},
            },
            mode="markers",
        ),
    )
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        template="plotly_white",
        # layout=go.Layout(xaxis={"showgrid": False}, yaxis={"showgrid": True}),
    )
    return fig
