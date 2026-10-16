"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    D = np.array(distances, dtype=float)
    
    # Leaves first, new internal nodes append to the end of the list/matrix
    nodes = [Node(name, None, 0.0, None, 0.0) for name in labels]
    counter = 0

    while len(nodes) > 2:
        n = len(nodes)
        r = D.sum(axis=1)

        # Q-Matrix: Q_ij = (n-2) * d_ij - r_i - r_j
        Q = (n - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf) # never join a node with itself

        i, j = np.unravel_index(np.argmin(Q), Q.shape)

        # Branch lenghts from joined pair to new node u
        d_iu = D[i, j] / 2 + (r[i] - r[j]) / (2 * (n-2))
        d_ju = D[i, j] - d_iu

        counter += 1
        u = Node(f"node_{counter}", nodes[i], d_iu, nodes[j], d_ju)

        # Distances from u to every remaining node k:
        keep = [k for k in range(n) if k not in (i,j)]
        d_u = (D[i, keep] + D[j, keep] - D[i,j]) / 2

        # Build the reduced matrix with u appeneded at the end
        newD = np.zeros((len(keep) + 1, len(keep) + 1))
        newD[:-1, :-1] = D[np.ix_(keep, keep)]
        newD[-1, :-1] = d_u
        newD[:-1, -1] = d_u
        D = newD
        nodes = [nodes[k] for k in keep] + [u]

    # Last two nodes 
    half = D[0,1] / 2
    return Node("root", nodes[0], half, nodes[1], half)

def plot_nj_tree(tree: Node, ax: Axes = None, root_stub=0.2, label_pad=0.25,
                 color="k", linewidth=1.5, fontsize=12,
                 xlabel="Genetic Distance", **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    import matplotlib.pyplot as plt

    if ax is None: 
        ax = plt.gca()

    next_leaf_y = 0 
    max_x = 0.0     # deepest leaf used to set x-limits 

    def line(xs, ys):
        ax.plot(xs, ys, color=color, lw=linewidth, solid_capstyle="projecting")

    def draw(node, x):
        """Draw the subtree below `node`, which sits at depth x. Returns node's y."""
        nonlocal next_leaf_y, max_x

        # leaf
        if node.left is None and node.right is None:
            y = next_leaf_y
            next_leaf_y += 1
            max_x = max(max_x, x)
            ax.text(x, y, f"  {node.name}", color=color, fontsize=fontsize,
                    verticalalignment="center")
            return y
        

        x_left = x + node.left_distance
        x_right = x + node.right_distance
        y_left = draw(node.left, x_left)
        y_right = draw(node.right, x_right)

        line([x, x_left], [y_left, y_left])         # horizontal: distance to left child
        line([x, x_right], [y_right, y_right])      # horizontal: distance to right child
        line([x, x], [y_left, y_right])             # vertical: joins the two children
        return (y_left + y_right) / 2
    

    y_root = draw(tree, 0.0)
    line([-root_stub, 0], [y_root, y_root])

    span = max_x + root_stub
    ax.set_xlim(-root_stub - 0.02 * span, max_x + label_pad * span)
    ax.set_ylim(-0.7, next_leaf_y - 0.3)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    def count_and_sort(node):
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1
        n_left = count_and_sort(node.left)
        n_right = count_and_sort(node.right)
        if n_left > n_right:
            node.left, node.right = node.right, node.left
            node.left_distance, node.right_distance = node.right_distance, node.left_distance
        return n_left + n_right

    count_and_sort(tree)

def hamming_matrix(seqs):
    """Pairwise Hamming distances (mismatch counts, gaps included) between aligned sequences."""
    # np.array raises a ValueError if the sequences are not all the same length
    arr = np.array([np.frombuffer(s.upper().encode(), dtype=np.uint8) for s in seqs])
    return (arr[:, None, :] != arr[None, :, :]).sum(axis=2).astype(float)


def find_leaf(node, text):
    """Return the first leaf whose name contains `text`, or None."""
    if node is None:
        return None
    if node.left is None and node.right is None:
        return node if text in node.name else None
    return find_leaf(node.left, text) or find_leaf(node.right, text)


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()