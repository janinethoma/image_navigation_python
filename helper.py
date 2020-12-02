from sklearn.neighbors import KDTree
import numpy as np
from math import sqrt
import random
import matplotlib.pyplot as plt


def get_edges(xy, max_dist, diff_to_path):
    """Get edges with maximal length max_dist from nx2 node coordinate array
    This method is used for sequences where a floor plan is not available."""

    n = xy.shape[0]
    tree = KDTree(xy)  # xy.shape() = (samples, sample_dimensions)
    nn, nn_dists = tree.query_radius(xy, r=max_dist, return_distance=True, sort_results=True)

    count = 0
    for i in range(n):
        count = count + len(nn_dists[i]) - 1

    num_edges = int(count / 2)

    edges = np.empty([2, num_edges], dtype=int)
    dists = np.empty(num_edges)

    # Generate edges
    e = 0
    for i in range(n):
        for j in range(1, len(nn[i])):
            if nn_dists[i][j] == 0:
                continue
            elif i < nn[i][j]:
                edges[:, e] = [i, nn[i][j]]
                dists[e] = nn_dists[i][j]
                e = e + 1

    edges = np.delete(edges, np.s_[e:num_edges], axis=1)
    dists = np.delete(dists, np.s_[e:num_edges])

    l = integrate_path(xy)
    to_delete = [i for i in range(len(dists)) if ((l[edges[1, i]] - l[edges[0, i]]) - dists[i]) > diff_to_path]
    print('Deleting {} edges, as they cut corners.'.format(len(to_delete)))
    edges = np.delete(edges, to_delete, axis=1)
    dists = np.delete(dists, to_delete)

    return edges, dists


def integrate_path(xy):
    """INTEGRATE_PATH(xy) where xy is a nx2 matrix of x and y coordinates"""

    dx = [sqrt((xy[i + 1, 0] - xy[i, 0]) ** 2 + (xy[i + 1, 1] - xy[i, 1]) ** 2) for i in range(xy.shape[0] - 1)]
    p_sum = [0] + dx

    for i in range(1, len(p_sum)):
        p_sum[i] = p_sum[i] + p_sum[i - 1]

    return p_sum


def greedy_anchors(xy, a_dist, a_nn):
    """
    Get anchor and anchor neighbourhood indices from nx2 xy locations
    """
    n = xy.shape[0]
    selected = [random.randrange(n)]
    remaining = np.setdiff1d(range(n), selected).tolist()
    while len(selected) < n:
        tree = KDTree(np.array([xy[i] for i in selected]))
        nn_dists, _ = tree.query([xy[i] for i in remaining], return_distance=True)

        i_max = np.argmax(nn_dists)

        if nn_dists[i_max] < (a_dist / 2):
            break

        selected = selected + [remaining[i_max]]
        remaining = np.delete(remaining, i_max)

    # Define anchor neighbourhood as anchor plus a_nn nearest neighbours
    selected.sort()
    tree = KDTree(np.array([xy[i] for i in remaining]))
    a_nbh = np.empty([len(selected), 1 + a_nn], int)
    for a in range(len(selected)):
        nn_i = tree.query([xy[selected[a]]], return_distance=False, k=a_nn)
        a_nbh[a, :] = [selected[a]] + [remaining[i] for i in nn_i[0]]

    return selected, a_nbh


def adjacent_edges(num_nodes, edges):
    """
    Get the outgoing and incoming edges for each node
    :param num_nodes: Number of nodes
    :param edges: 2xn array of directed edges
    :return: list of list of outgoing nodes, list of list of incoming nodes
    """
    outgoing = [[] for i in range(num_nodes)]
    incoming = [[] for i in range(num_nodes)]

    for i in range(len(edges[0])):
        outgoing[edges[0][i]].append(i)
        incoming[edges[1][i]].append(i)

    return outgoing, incoming


def sensitivity(edges, f_dists):
    """
    Return sensitivity for network flow algorithm
    :param edges: 2xn array of directed edges
    :param f_dists: n feature distances
    :return: n sensitivity values
    """

    nodes = np.arange(max(edges[0]) + 1)
    # It is unlikely, that a node has no edges.
    # For data with many geographic outliers use
    # nodes = list(set(edges[0]))
    # to get unique nodes (and adjust code below)
    denominator = np.zeros(nodes.shape)
    for i, d in enumerate(f_dists):
        denominator[edges[0][i]] = denominator[edges[0][i]] + d

    return 1 - f_dists / denominator[edges[0]]


def huber(v, t):
    if abs(v) <= t:
        return (v ** 2.0) / 2.0
    else:
        return t * (abs(v) - t / 2.0)


def plot_accuracy_vs_distance(dist_geom, idx_match, label=None, color='green', linestyle='dashed'):
    all_dist = np.zeros(len(dist_geom))

    for i in range(dist_geom.shape[0]):
        idx_i = idx_match[i, :]
        all_dist[i] = np.min(dist_geom[i, idx_i])

    dist_sorted = np.sort(all_dist)
    max_val = np.max(dist_sorted)
    dist_val = np.arange(max_val / 100.0, max_val, max_val / 100.0)
    N = len(dist_val)
    acc_val = np.zeros(N)

    for i in range(N):
        acc_val[i] = 100 * sum(dist_sorted <= dist_val[i]) / len(all_dist)

    plt.plot(dist_val, acc_val, label=label,  color=color, linestyle=linestyle)

