import numpy as np
import math
from helper import huber, adjacent_edges
import mosek.fusion as mf
from numpy import matlib


def match_with_flow(query_xy, ref_xy, visual_dist, topN=0):
    """
    Match a series of query images to reference images based on visual distances
    :param query_xy: nx2 query locations
    :param ref_xy: mx2 reference locations
    :param visual_dist: mxn visual distances
    :param topN: 0 to generate e_il between all v'/p combinations, n to only generate edges between v'/p and their
    top-n visually closest p/v'
    :return: matched reference indices
    """

    query_xy = np.array(query_xy).transpose()  # 2xn
    ref_xy = np.array(ref_xy).transpose()  # 2xm
    visual_dist = np.array(visual_dist).transpose()  # nxm

    dist_bound_init = 30

    t = np.mean(ref_xy, 1)
    s = np.mean(np.sqrt(np.sum(np.square(ref_xy - np.matlib.repmat(t, ref_xy.shape[1], 1).transpose()), 0)))
    dist_bound_norm = dist_bound_init / s

    norm_mat = math.sqrt(2) * np.array([[1.0 / s, 0.0], [0.0, 1.0 / s]])
    query_xy = norm_mat.dot((query_xy - np.matlib.repmat(t, query_xy.shape[1], 1).transpose()))
    ref_xy = norm_mat.dot((ref_xy - np.matlib.repmat(t, ref_xy.shape[1], 1).transpose()))

    visual_dist = visual_dist / np.median(np.min(visual_dist, 0))

    idx_of_nth_closest_v = np.argsort(visual_dist, 1)
    dist_2_nth_closest_v = np.sort(visual_dist, 1)

    # Initialization
    v_xy = ref_xy
    num_p = query_xy.shape[1]  # 145
    num_v = v_xy.shape[1]  # 100
    T = num_p
    cost_thresh = np.median(dist_2_nth_closest_v[:, 0])

    ###################################################################################################################
    #  S to V', s=1
    ###################################################################################################################
    node_idx = 0
    arc_cap = [num_p] * num_v
    arc_i = [0] * num_v
    arc_j = [i for i in range(1, num_v + 1)]
    arc_base = [0] * num_v

    ###################################################################################################################
    # form V' to P
    ###################################################################################################################
    node_idx = node_idx + 1
    one_dir_start = len(arc_i)

    # When matching long sequences, it may be computationally beneficial to only generate edges between v'/p and
    # their top-n visually closest p/v' instead of all v'-p combinations.
    if topN == 0:  # Generate all p-v' edges
        print('Generating all v-p edges.')
        for i in range(num_v):
            for j in range(num_p):
                arc_i = arc_i + [node_idx + i]
                arc_j = arc_j + [node_idx + num_v + j]
                arc_base = arc_base + [huber(visual_dist[j, i], cost_thresh)]
                arc_cap = arc_cap + [num_p]

    else:  # Generate fewer edges
        print('Generating visually close v-p edges.')
        match_length = np.min([num_p, num_v, topN])
        idx_of_nth_closest_p = np.argsort(visual_dist, 0)
        dist_2_nth_closest_p = np.sort(visual_dist, 0)
        for i in range(num_v):
            for j in range(min(match_length, num_p)):
                arc_i = arc_i + [node_idx + i]
                arc_j = arc_j + [node_idx + num_v + idx_of_nth_closest_p[j, i]]
                arc_base = arc_base + [huber(dist_2_nth_closest_p[j, i], cost_thresh)]
                arc_cap = arc_cap + [num_p]
        for i in range(num_p):
            for j in range(min(match_length, num_v)):
                arc_i = arc_i + [node_idx + idx_of_nth_closest_v[i, j]]
                arc_j = arc_j + [node_idx + num_v + i]
                arc_base = arc_base + [huber(dist_2_nth_closest_v[i, j], cost_thresh)]
                arc_cap = arc_cap + [num_p]

    one_dir_end = len(arc_i)
    node_idx = node_idx + num_v

    ###################################################################################################################
    #  from P to T
    ###################################################################################################################
    for i in range(num_p):
        arc_i = arc_i + [node_idx + i]
        arc_j = arc_j + [node_idx + num_p]
        arc_base = arc_base + [0]  # no cost
        arc_cap = arc_cap + [1]  # limited capacity

    ###################################################################################################################
    # Mosek optimization
    ###################################################################################################################
    num_all_v = 1 + num_v + num_p + 1  # 1 source, 1 sink, num_v, num_p

    narcs = len(arc_i)
    M = mf.Model('Sequence matching model')
    x = M.variable('x', narcs, mf.Domain.inRange(0, arc_cap))
    y = M.variable('y', narcs, mf.Domain.inRange(0, arc_cap))

    M.objective('Matching objective', mf.ObjectiveSense.Minimize,
                mf.Expr.add(mf.Expr.dot(arc_base, x), mf.Expr.dot(arc_base, y)))

    outgoing, incoming = adjacent_edges(num_all_v, [arc_i, arc_j])

    for idx in range(num_all_v):  # Iterate over all vertices
        v = 0
        if outgoing[idx]:
            v = mf.Expr.sub(mf.Expr.sum(x.pick(outgoing[idx])), mf.Expr.sum(y.pick(outgoing[idx])))

        if incoming[idx]:
            v = mf.Expr.add(v, mf.Expr.sub(mf.Expr.sum(y.pick(incoming[idx])), mf.Expr.sum(x.pick(incoming[idx]))))

        if not outgoing[idx] + incoming[idx]:
            continue

        if idx == 0:  # source
            M.constraint(v, mf.Domain.equalsTo(T))
        elif idx == num_all_v - 1:
            M.constraint(v, mf.Domain.equalsTo(-T))
        else:
            M.constraint(v, mf.Domain.equalsTo(0))

    # Geometric constrains
    _, incoming_in_range = adjacent_edges(num_all_v, [arc_i[one_dir_start:one_dir_end],
                                                      arc_j[one_dir_start:one_dir_end]])

    for i in range(num_v + 1, num_v + num_p):
        selected1 = incoming_in_range[i]
        selected2 = incoming_in_range[i + 1]

        if not (selected1 and selected2):
            continue

        flow_idx1 = np.array(selected1) + one_dir_start
        flow_idx2 = np.array(selected2) + one_dir_start

        vertex_idx1 = np.array([arc_i[i + one_dir_start] - 1 for i in selected1])
        vertex_idx2 = np.array([arc_i[i + one_dir_start] - 1 for i in selected2])

        v1_all = mf.Matrix.dense(v_xy[:, vertex_idx1])
        v2_all = mf.Matrix.dense(v_xy[:, vertex_idx2])

        x1_all = x.pick(flow_idx1.tolist()).transpose()
        x2_all = x.pick(flow_idx2.tolist()).transpose()

        fx1 = mf.Expr.sum(mf.Expr.mulElm(mf.Expr.vstack(x1_all, x1_all), v1_all), 1)
        fx2 = mf.Expr.sum(mf.Expr.mulElm(mf.Expr.vstack(x2_all, x2_all), v2_all), 1)

        M.constraint(mf.Expr.vstack(dist_bound_norm, mf.Expr.sub(fx1, fx2)), mf.Domain.inQCone())

    M.solve()
    flow = x.level()
    M.dispose()

    # Compute matches from flow
    match_final_idx = np.zeros([num_p, 1], int)
    for i in range(num_v + 1, num_v + num_p + 1):
        selected1 = incoming_in_range[i]
        flow_idx1 = np.array(selected1) + num_v
        vertex_idx1 = np.array([arc_i[i + num_v] - 1 for i in selected1])

        v1_all = v_xy[:, vertex_idx1]  # Adjacent landmark locations
        x1_all = flow[flow_idx1]  # Flow from adjacent landmark locations

        Y = np.sum(np.multiply(x1_all, v1_all), 1)  # XL for error computation
        id_min = np.argmin(np.sum((ref_xy - np.matlib.repmat(Y, num_v, 1).transpose()) ** 2, 0))  # closest landmark
        match_final_idx[i - (num_v + 1)] = id_min

    return match_final_idx
