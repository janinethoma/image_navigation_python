"""
Demo for Mapping, Localization and Path Planning for Image-based Navigation using Visual Features and Map

This demo does the following using the concepts introduced in our paper:
1) Find landmarks in a sub sequence of the Oxford Robotcar run from  2015-10-29 12:18:17
2) Match a short query sequence from 2014-11-18 13:20:12
Both, reference and query, had to be shortened due to supplementary
material data limis.

If you do not have mosek installed, you can have a look at the saved
figures in the results folder instead.

Please adjust MOSEK import path to match your installation.
"""

import pickle
import matplotlib.pyplot as plt
import os
from helper import get_edges, plot_accuracy_vs_distance
from sampling import sample_with_flow, sample_uniformly
from matching import match_with_flow
import numpy as np
from scipy.spatial.distance import cdist


def main():
    ###################################################################################################################
    # Load image locations and precalculated NetVLAD image feature distances
    ###################################################################################################################

    print('Building topology')
    with open('data.pickle', 'rb') as f:
        f_dists_query_ref, f_dists_ref_ref, query, ref = pickle.load(f)

    ref['x'] = ref['xy'][:, 0]
    ref['y'] = ref['xy'][:, 1]

    query['x'] = query['xy'][:, 0]
    query['y'] = query['xy'][:, 1]

    plt.figure()
    plt.scatter(ref['x'], ref['y'], marker='.', c='000000', linewidths=0)
    plt.scatter(query['x'], query['y'], c=query['t'].flatten(), marker='*')
    plt.legend({'Reference', 'Query'})
    plt.title('Reference and query sequence')
    plt.xlabel('UTM Easting [m]')
    plt.ylabel('UTM Northing [m]')
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'original_sequences.pdf'))
    plt.show()

    # Generate topology
    edges, geo_dists = get_edges(ref['xy'], 10, 2)

    source_idx = [0]  # This should always be a list
    sink_idx = [len(ref['x']) - 1]

    plt.figure()
    for i in range(0, len(geo_dists), 50):
        p1 = ref['xy'][edges[0, i], :]
        p2 = ref['xy'][edges[1, i], :]
        plt.plot([p2[0], p1[0]], [p2[1], p1[1]], label=None, linewidth=0.1)

    plt.scatter(ref['x'][source_idx], ref['y'][source_idx], c='none', marker='o', label='Source', edgecolors='r')
    plt.scatter(ref['x'][sink_idx], ref['y'][sink_idx], c='none', marker='o', label='Sink', edgecolors='g')
    plt.legend()
    plt.title('Topology')
    plt.xlabel('UTM Easting [m]')
    plt.ylabel('UTM Northing [m]')
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'topology.pdf'))
    plt.show()

    ###################################################################################################################
    # Sample landmarks with flow
    ###################################################################################################################

    print('Sampling landmarks')
    if os.path.isfile('temp.pickle'):
        print('Loading precalculated landmarks. Delete temp.pickle, if you want to recalculate.')
        with open('temp.pickle', 'rb') as f:
            num_landmarks, feat_dists, flow_lm, uniform_lm = pickle.load(f)
    else:

        num_landmarks = 150

        # Get feature distance for each edge in reference topology
        feat_dists = [f_dists_ref_ref[edges[0, i]][edges[1, i]] for i in range(edges.shape[1])]

        # Get landmarks
        flow_lm = sample_with_flow(ref['xy'], edges, source_idx, sink_idx, geo_dists, feat_dists, num_landmarks)
        uniform_lm = sample_uniformly(ref['xy'], num_landmarks)

        plt.figure()
        plt.scatter(ref['x'], ref['y'], marker='.', c='k', linewidths=0, label=None)
        plt.scatter(ref['x'][flow_lm], ref['y'][flow_lm], c='none', marker='o',
                    label='Flow landmarks', edgecolors='b', linewidth=0.1)
        plt.scatter(ref['x'][uniform_lm], ref['y'][uniform_lm], c=np.linspace(0, 1, len(uniform_lm)),
                    marker='*', label='Uniform landmarks', linewidth=0.1)
        plt.legend()
        plt.title('Selected landmarks')
        plt.xlabel('UTM Easting [m]')
        plt.ylabel('UTM Northing [m]')
        plt.tight_layout()
        plt.savefig(os.path.join('results', 'landmarks.pdf'))
        plt.show()

        with open('temp.pickle', 'wb') as f:
            pickle.dump([num_landmarks, feat_dists, flow_lm, uniform_lm], f)

    ###################################################################################################################
    # Match with flow
    ###################################################################################################################

    print('Matching sequences')
    F = f_dists_query_ref
    D = cdist(query['xy'], ref['xy'])  # num_query x num_ref

    # Match with flow
    flow_matches = match_with_flow(query['xy'], [ref['xy'][i] for i in flow_lm], F[flow_lm, :], topN=0)

    # Retrieve without matching
    feature_matches = np.argsort(F[uniform_lm, :].transpose(), 1)
    lm_only_matches = np.argsort(F[flow_lm, :].transpose(), 1)

    # Display accuracy
    plt.figure()
    plot_accuracy_vs_distance(D[:, uniform_lm], feature_matches[:, 0:10], color=(0, 0.4470, 0.7410), linestyle=':',
                              label='Top-10 uniform landmarks, no matching')
    plot_accuracy_vs_distance(D[:, flow_lm], flow_matches, color=(0.4660, 0.6740, 0.1880), linestyle='-',
                              label='Our landmarks, our matching')
    plot_accuracy_vs_distance(D[:, flow_lm], lm_only_matches[:, 0:1], color=(0, 0, 0), linestyle=':',
                              label='Our landmarks, no matching')
    plot_accuracy_vs_distance(D[:, uniform_lm], feature_matches[:, 0:1], color=(0.8500, 0.3250, 0.0980), linestyle=':',
                              label='Top-1 uniform landmarks, no matching')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.title('Accuracy vs. distance')
    plt.xlabel('Distance [m]')
    plt.ylabel('Accuracy [m]')
    plt.tight_layout()
    plt.savefig(os.path.join('results', 'accuracy.pdf'))
    plt.show()


if __name__ == '__main__':
    main()
