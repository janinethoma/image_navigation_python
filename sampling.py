import statistics
from helper import greedy_anchors, sensitivity, adjacent_edges, integrate_path
import numpy as np
import mosek.fusion as mf


def sample_with_flow(xy, edges, source_idx, sink_idx, geo_dists, feat_dists, num_to_choose):
    """
    Finds num_to_choose landmarks using network flow.
    :param xy: nx2 node locations
    :param edges: 2xm edges represented as pair of node indices
    :param source_idx: list of indices of source nodes
    :param sink_idx: list of indices of sink nodes
    :param geo_dists: m length of edges
    :param feat_dists: m feat distances between vertices connected by edge
    :param num_to_choose: number of landmarks to select
    :return: node indices of selected landmarks
    """

    T = 0.1  # Total flow
    a_dist = 4  # Distance between anchors
    a_nn = 5  # Number of images in anchor nbh
    tg = 0.1  # Flow through anchor nbh

    # Get costs
    feat_dists = feat_dists / statistics.median(feat_dists)
    costs = 1. / (1e-6 + feat_dists)

    # Get anchors
    anchors, a_nbh = greedy_anchors(xy, a_dist, a_nn)

    # Get capacities
    caps = geo_dists

    # Source & sink capacity
    special_idx = source_idx + sink_idx
    caps = [(T if ((edges[0][i] in special_idx) or (edges[1][i] in special_idx))
             else caps[i]) for i in range(len(caps))]

    # Convert to directed graph
    arc_cap = caps + caps
    arc_base = costs.tolist() + costs.tolist()
    arc_i = list(edges[0]) + list(edges[1])
    arc_j = list(edges[1]) + list(edges[0])

    n = len(xy)
    narcs = len(arc_i)
    print('The number of nodes is {}.'.format(n))
    print('The number of edges is {}.'.format(narcs))

    # Get sensitivity
    # This should be done with directed graph
    arc_sens = sensitivity([arc_i, arc_j], list(feat_dists) + list(feat_dists))

    # Mosek code
    print('Starting optimization')
    M = mf.Model('LandmarkSelectionModel')
    f = M.variable('f', narcs, mf.Domain.inRange(0, arc_cap))  # Flow per edge
    z = M.variable('s', narcs, mf.Domain.greaterThan(0))  # Additional cost term

    # Edge lookup for faster calculations
    outgoing, incoming = adjacent_edges(n, [arc_i, arc_j])

    # Set the objective:
    M.objective('Minimize total cost', mf.ObjectiveSense.Minimize,
                mf.Expr.add(mf.Expr.dot(arc_base, f), mf.Expr.sum(z)))

    # Flow conservation constraints
    for idx in range(n):  # For each node
        f_tot = 0  # Total flow
        if outgoing[idx]:  # Node has outgoing edges (empty lists are false)
            out_picks = f.pick(outgoing[idx])
            f_out = mf.Expr.sum(out_picks)

        if incoming[idx]:  # Incoming edges of node idx
            in_picks = f.pick(incoming[idx])
            f_in = mf.Expr.sum(in_picks)
            f_tot = mf.Expr.sub(f_out, f_in)

        if not outgoing[idx] + incoming[idx]:
            continue

        if idx in source_idx:
            M.constraint(f_tot, mf.Domain.equalsTo(T))
        elif idx in sink_idx:
            M.constraint(f_tot, mf.Domain.equalsTo(-T * (float(len(source_idx)) / float(len(sink_idx)))))
        else:
            M.constraint(f_tot, mf.Domain.equalsTo(0))

    # Anchor constraint, i.e.geometric representation
    for a in range(len(anchors)):
        all_adj_edges = []
        for nn in a_nbh[a, :]:  # Each node in given anchor nbh
            all_adj_edges = all_adj_edges + outgoing[nn] + incoming[nn]
        all_adj_edges = list(set(all_adj_edges))  # Find unique edges
        M.constraint(mf.Expr.sum(f.pick(all_adj_edges)), mf.Domain.greaterThan(tg))

    # Visual representation
    # Rotated quadratic cone = 2 * lhs1 * lhs2 > rhs ^ 2
    lhs1 = mf.Expr.mul(0.5, mf.Expr.sub(arc_cap, f))
    lhs2 = mf.Expr.mulElm(z, (1. / (np.array(arc_sens) * np.array(arc_cap))))
    stack = mf.Expr.hstack(lhs1, lhs2, f)
    M.constraint(stack, mf.Domain.inRotatedQCone().axis(2))  # Each row is in a rotated quadratic cone

    print('Set constraints. Solving.')

    M.solve()
    flow = f.level()
    M.dispose()

    # Choose nodes with highest flow & implicitly choose tau to get desired number of landmarks
    node_flow = np.zeros(n)
    for i, d in enumerate(flow):
        node_flow[arc_i[i]] = node_flow[arc_i[i]] + d
    flow_sorted_idx = np.argsort(node_flow)
    highest_idx = flow_sorted_idx[-num_to_choose:]

    return np.sort(highest_idx).tolist()  # Return sorted list of landmarks


def sample_uniformly(xy, num_to_choose):
    """
    Get num_to_choose uniformly sampled landmarks
    :param xy: nx2 node locations
    :param num_to_choose: number of landmarks to select
    :return: node indices of selected landmarks
    """
    l = integrate_path(xy)
    locs = np.linspace(0, l[-1], num_to_choose)
    return [np.argmin(abs(l-loc)) for loc in locs]
