"""A small MLP built from scratch in NumPy — no PyTorch, no sklearn.

Architecture: 8 -> 24 (ReLU) -> 4 (softmax).  The parameter dict is the
unit that travels between server and clients in federated learning, so
everything here is expressed as plain ``{name: ndarray}`` dicts: that is
what gets averaged (FedAvg), sparsified (top-k) and noised (DP).
"""
from __future__ import annotations

import numpy as np

INPUT_DIM = 8
HIDDEN_DIM = 24
OUTPUT_DIM = 4


def init_params(seed: int = 0) -> dict:
    """He-initialised weights, zero biases."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / INPUT_DIM),
                         size=(INPUT_DIM, HIDDEN_DIM)),
        "b1": np.zeros(HIDDEN_DIM),
        "W2": rng.normal(0.0, np.sqrt(2.0 / HIDDEN_DIM),
                         size=(HIDDEN_DIM, OUTPUT_DIM)),
        "b2": np.zeros(OUTPUT_DIM),
    }


def forward(params: dict, X: np.ndarray):
    """Return (probabilities, cache) for a forward pass."""
    X = np.asarray(X, dtype=float)
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.maximum(0.0, z1)
    z2 = a1 @ params["W2"] + params["b2"]
    z2 = z2 - z2.max(axis=1, keepdims=True)
    exp_z = np.exp(z2)
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)
    cache = {"X": X, "z1": z1, "a1": a1, "probs": probs}
    return probs, cache


def loss_and_grads(params: dict, X, y) -> tuple:
    """Cross-entropy loss and gradients for one batch."""
    n = len(y)
    probs, cache = forward(params, X)
    y = np.asarray(y)
    loss = -float(np.log(probs[np.arange(n), y] + 1e-12).mean())
    dz2 = probs.copy()
    dz2[np.arange(n), y] -= 1.0
    dz2 /= n
    grads = {
        "W2": cache["a1"].T @ dz2,
        "b2": dz2.sum(axis=0),
        "W1": cache["X"].T @ ((dz2 @ params["W2"].T)
                              * (cache["z1"] > 0)),
        "b1": (((dz2 @ params["W2"].T) * (cache["z1"] > 0)).sum(axis=0)),
    }
    return loss, grads


def sgd_step(params: dict, grads: dict, lr: float) -> dict:
    """One SGD step; returns a new parameter dict (inputs untouched)."""
    return {k: params[k] - lr * grads[k] for k in params}


def train_batch(params: dict, X, y, lr: float = 0.05,
                epochs: int = 1, batch_size: int = 32,
                seed: int = 0, prox_params: dict | None = None,
                prox_mu: float = 0.0) -> dict:
    """Train ``params`` on (X, y) for ``epochs`` full passes.

    ``prox_params``/``prox_mu`` implement the FedProx proximal term:
    a penalty ``(mu/2) * ||w - w_global||^2`` that keeps a client from
    drifting too far from the global model on skewed local data.
    """
    rng = np.random.default_rng(seed)
    params = {k: v.copy() for k, v in params.items()}
    n = len(y)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            _, grads = loss_and_grads(params, X[idx], y[idx])
            if prox_params is not None and prox_mu > 0.0:
                for k in grads:
                    grads[k] = grads[k] + prox_mu * (params[k] - prox_params[k])
            params = sgd_step(params, grads, lr)
    return params


def predict(params: dict, X) -> np.ndarray:
    probs, _ = forward(params, X)
    return probs.argmax(axis=1)


def accuracy(params: dict, X, y) -> float:
    return float((predict(params, X) == np.asarray(y)).mean())


def param_delta(new: dict, old: dict) -> dict:
    """The update a client sends: new - old."""
    return {k: new[k] - old[k] for k in old}


def apply_delta(params: dict, delta: dict) -> dict:
    return {k: params[k] + delta[k] for k in params}


def average_params(param_list: list, weights: list | None = None) -> dict:
    """Weighted average of parameter dicts (FedAvg core)."""
    if not param_list:
        raise ValueError("cannot average an empty list of parameters")
    if weights is None:
        weights = [1.0 / len(param_list)] * len(param_list)
    total = float(sum(weights))
    weights = [w / total for w in weights]
    out = {}
    for k in param_list[0]:
        out[k] = sum(w * p[k] for w, p in zip(weights, param_list))
    return out


def num_params(params: dict) -> int:
    return int(sum(v.size for v in params.values()))


def sparsify_topk(delta: dict, keep_fraction: float) -> dict:
    """Keep only the largest-magnitude ``keep_fraction`` of each tensor
    (communication-efficient updates); the rest are zeroed, not dropped,
    so shapes stay identical and averaging still works."""
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if keep_fraction == 1.0:
        return {k: v.copy() for k, v in delta.items()}
    out = {}
    for k, v in delta.items():
        flat = np.abs(v).ravel()
        keep = max(1, int(round(keep_fraction * flat.size)))
        if keep >= flat.size:
            out[k] = v.copy()
            continue
        threshold = np.partition(flat, flat.size - keep)[flat.size - keep]
        out[k] = np.where(np.abs(v) >= threshold, v, 0.0)
    return out


def add_gaussian_noise(delta: dict, sigma: float, seed: int = 0) -> dict:
    """Differential-privacy style noise on an update (Gaussian mechanism)."""
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0.0:
        return {k: v.copy() for k, v in delta.items()}
    rng = np.random.default_rng(seed)
    return {k: v + rng.normal(0.0, sigma, size=v.shape) for k, v in delta.items()}
