"""Synthetic classification data and non-IID federated partitions.

The task is a 4-class problem in 8 dimensions: each class is a Gaussian
blob around its own centre.  Data is synthetic, so no real user data is
ever involved — the point of the simulation is the *partitioning* and the
*protocol*, not the dataset.

Non-IID partitioning (the realistic federated case) is label skew: every
client has two dominant classes that supply most of its data, with the
remaining classes present only in a small share.  An ``iid`` mode is kept
for comparison, so the simulation can show exactly how much non-IID data
costs federated training.
"""
from __future__ import annotations

import numpy as np

N_FEATURES = 8
N_CLASSES = 4


def make_centres(seed: int = 0) -> np.ndarray:
    """Class centres, well separated so a small MLP can learn the task."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 2.2, size=(N_CLASSES, N_FEATURES))
    # Push the centres apart a little more for a clean, learnable task.
    centres[1] += np.array([3.0, 0, 0, 0, 0, 0, 0, 0])
    centres[2] += np.array([0, 3.0, 0, 0, 0, 0, 0, 0])
    centres[3] += np.array([0, 0, 3.0, 0, 0, 0, 0, 0])
    return centres


def generate_data(n_train: int = 2000, n_test: int = 400, seed: int = 0,
                  noise: float = 1.8):
    """Balanced synthetic train/test split. Returns (Xtr, ytr, Xte, yte)."""
    rng = np.random.default_rng(seed)
    centres = make_centres(seed)
    def _make(n):
        y = np.repeat(np.arange(N_CLASSES), n // N_CLASSES)
        # Any remainder goes to class 0, then shuffle.
        y = np.concatenate([y, np.zeros(n - len(y), dtype=int)])
        X = centres[y] + rng.normal(0.0, noise, size=(n, N_FEATURES))
        perm = rng.permutation(n)
        return X[perm], y[perm]
    X_train, y_train = _make(n_train)
    X_test, y_test = _make(n_test)
    return X_train, y_train, X_test, y_test


def partition_noniid(X, y, n_clients: int = 10, seed: int = 0,
                     dominant_share: float = 0.8):
    """Label-skew partition.

    Client ``i`` has dominant classes ``(2i % 4, 2i+1 % 4)``: ``dominant_share``
    of its data comes from those two classes, the rest is spread over the
    other two.  Every client still sees every class (a little), which is
    what makes local training possible at all — but the skew is strong
    enough that a locally-trained model is badly biased.
    """
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    if not 0.0 <= dominant_share <= 1.0:
        raise ValueError("dominant_share must be in [0, 1]")
    rng = np.random.default_rng(seed)
    n = len(y)
    per_client = n // n_clients
    # Shuffle the class pools first so assignment is not ordered.
    pools = {}
    for c in range(N_CLASSES):
        idx = np.where(y == c)[0]
        pools[c] = list(rng.permutation(idx))
    clients_idx = []
    for i in range(n_clients):
        dom = [(2 * i) % N_CLASSES, (2 * i + 1) % N_CLASSES]
        other = [c for c in range(N_CLASSES) if c not in dom]
        n_dom = int(round(per_client * dominant_share))
        n_other = per_client - n_dom
        take = []
        # Split the dominant share evenly over the two dominant classes.
        for j, c in enumerate(dom):
            k = n_dom // 2 + (n_dom % 2 if j == 0 else 0)
            take.extend(pools[c][:k])
            pools[c] = pools[c][k:]
        for j, c in enumerate(other):
            k = n_other // 2 + (n_other % 2 if j == 0 else 0)
            take.extend(pools[c][:k])
            pools[c] = pools[c][k:]
        clients_idx.append(list(take))
    # Any samples left in the pools (a class ran out for a later client
    # when n_clients does not divide the dominant-class cycle evenly)
    # are dealt round-robin so no data is ever dropped.
    leftover = [idx for v in pools.values() for idx in v]
    if leftover:
        rng.shuffle(leftover)
        for j, idx in enumerate(leftover):
            clients_idx[j % n_clients].append(idx)
    clients = []
    for take in clients_idx:
        take = np.array(take, dtype=int)
        rng.shuffle(take)
        clients.append((X[take], y[take]))
    return clients


def partition_iid(X, y, n_clients: int = 10, seed: int = 0):
    """Uniform random partition (the easy, unrealistic case)."""
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    splits = np.array_split(perm, n_clients)
    return [(X[s], y[s]) for s in splits]


def class_histogram(y) -> dict:
    """Class counts as a plain dict, for reports and tests."""
    counts = np.bincount(np.asarray(y), minlength=N_CLASSES)
    return {int(c): int(counts[c]) for c in range(N_CLASSES)}


def partition_skew(clients) -> float:
    """Mean total-variation distance of client label distributions from
    the global one — 0 is IID, higher is more skewed."""
    all_y = np.concatenate([y for _, y in clients])
    global_hist = np.bincount(all_y, minlength=N_CLASSES) / len(all_y)
    dists = []
    for _, y in clients:
        h = np.bincount(y, minlength=N_CLASSES) / max(len(y), 1)
        dists.append(0.5 * float(np.abs(h - global_hist).sum()))
    return float(np.mean(dists))
