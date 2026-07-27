# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Hierarchical IVF drill-down over unit-normalized memory keys. A coarse quantizer
#   (spherical k-means, n_cells ~ N/cell_target) partitions the keys; search scores the
#   centroids, probes the top-nprobe cells, and exact-scores only those candidates.
#   Candidates scored ~ nprobe * N/n_cells, sub-linear in N.
# - Pure numpy, no FAISS dependency. One concern: cell partition + probe-and-rank. The
#   keys stay the source of truth; this only prunes what gets scored.
# veritate_core/memory/hindex.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np

# ------------------------------------------------------------------------------------
# Constants

KMEANS_ITERS = 12
CELL_TARGET  = 256     # target keys per cell -> n_cells = N / CELL_TARGET
EPS          = 1e-8
KEY_DTYPE    = np.float32


# ------------------------------------------------------------------------------------
# Functions

def _spherical_kmeans(x, k, iters, rng):
    """Cosine k-means on unit-norm rows. Returns (centroids [k,D], assign [N])."""
    c = x[rng.choice(x.shape[0], k, replace=False)].copy()
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        c = np.zeros((k, x.shape[1]), dtype=KEY_DTYPE)
        np.add.at(c, a, x)
        c /= (np.linalg.norm(c, axis=1, keepdims=True) + EPS)
    return c, (x @ c.T).argmax(1)


class HIndex:
    def __init__(self, keys, cell_target=CELL_TARGET, iters=KMEANS_ITERS, seed=0):
        self.keys = keys.astype(KEY_DTYPE)
        k = max(1, keys.shape[0] // cell_target)
        rng = np.random.default_rng(seed)
        self.centroids, assign = _spherical_kmeans(self.keys, k, iters, rng)
        order = np.argsort(assign, kind="stable")
        bounds = np.searchsorted(assign[order], np.arange(k + 1))
        self.cells = [order[bounds[j]:bounds[j + 1]] for j in range(k)]

    def n_cells(self):
        return len(self.centroids)

    def search(self, q, topk, nprobe):
        """(top indices, scores, candidates_scored). candidates_scored counts the
        centroid scan plus the probed-cell keys, the true sub-linear read cost."""
        q = q.astype(KEY_DTYPE)
        cell_scores = self.centroids @ q
        p = min(nprobe, cell_scores.shape[0])
        probe = np.argpartition(-cell_scores, p - 1)[:p]
        cand = np.concatenate([self.cells[c] for c in probe])
        scored = int(self.centroids.shape[0] + cand.shape[0])
        if cand.shape[0] == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=KEY_DTYPE), scored
        sc = self.keys[cand] @ q
        t = min(topk, sc.shape[0])
        top = cand[np.argpartition(-sc, t - 1)[:t]]
        top = top[np.argsort(-(self.keys[top] @ q))]
        return top, self.keys[top] @ q, scored
