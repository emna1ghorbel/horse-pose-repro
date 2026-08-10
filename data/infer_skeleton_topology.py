"""
Version corrigée de l'inférence de topologie : utilise la distance MOYENNE
(proximité spatiale) entre points comme critère pour l'arbre couvrant
minimal, plutôt que le coefficient de variation seul (qui créait des hubs
artificiels autour de points peu mobiles comme la tête).

Principe : deux points reliés par un os sont, en anatomie normale, proches
l'un de l'autre (un sabot est proche du genou, pas de l'oeil) - c'est un
critère plus robuste et plus intuitif que la seule variance.

Usage :
    python infer_skeleton_topology_v2.py --input-dir ./raw/horse_synthetic/horse_combineds5r5_texture --n-samples 2000
"""
import argparse
import glob
import os
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

HORSE_KEYPOINT_INDICES = np.array([
    1718, 1684, 1271, 1634, 1650, 1643, 1659, 925, 392, 564,
    993, 726, 1585, 1556, 427, 1548, 967, 877
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--pose-filter", type=str, default="idle",
                         help="Filtrer sur des poses calmes pour un signal plus fiable")
    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.input_dir, "*kpts.npy")))
    filtered = [f for f in all_files if args.pose_filter.lower() in os.path.basename(f).lower()]
    if len(filtered) < 50:
        print(f"Peu de fichiers filtrés sur '{args.pose_filter}' ({len(filtered)}), utilisation de tous les fichiers.")
        filtered = all_files

    idx = np.random.choice(len(filtered), min(args.n_samples, len(filtered)), replace=False)
    sample_files = [filtered[i] for i in idx]

    n_joints = len(HORSE_KEYPOINT_INDICES)
    all_pts = []
    for f in sample_files:
        pts = np.load(f).astype(np.float32)[HORSE_KEYPOINT_INDICES]
        all_pts.append(pts[:, :2])
    all_pts = np.stack(all_pts)  # (n_samples, 18, 2) -- coordonnées BRUTES (pixels), non normalisées

    print(f"Poses utilisées pour l'analyse : {all_pts.shape[0]}")

    # Distance moyenne (sur coordonnées brutes) entre chaque paire de points
    dist_matrix = np.zeros((all_pts.shape[0], n_joints, n_joints))
    for i in range(n_joints):
        for j in range(n_joints):
            dist_matrix[:, i, j] = np.linalg.norm(all_pts[:, i, :] - all_pts[:, j, :], axis=1)

    mean_dist = dist_matrix.mean(axis=0)  # (18, 18)
    np.fill_diagonal(mean_dist, np.inf)

    print(f"Distance moyenne min (hors diagonale) : {mean_dist[mean_dist != np.inf].min():.2f}")
    print(f"Distance moyenne max : {mean_dist[mean_dist != np.inf].max():.2f}")

    # MST sur la distance moyenne : les connexions les plus courtes = probables "os"
    graph = csr_matrix(mean_dist)
    mst = minimum_spanning_tree(graph)
    mst_array = mst.toarray()

    edges = []
    for i in range(n_joints):
        for j in range(n_joints):
            if mst_array[i, j] > 0:
                edges.append((i, j, mst_array[i, j]))
    edges.sort(key=lambda e: e[2])

    print(f"\n=== Topologie déduite par proximité spatiale ({len(edges)} arêtes) ===")
    for i, j, d in edges:
        print(f"  {i:2d} -- {j:2d}   (distance moyenne={d:.2f})")

    edge_list_str = "[" + ", ".join(f"({i}, {j})" for i, j, _ in edges) + "]"
    print(f"\nÀ copier dans SKELETON_EDGES :\n{edge_list_str}")


if __name__ == "__main__":
    main()