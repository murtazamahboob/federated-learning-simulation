"""The client side of federated learning.

A client holds its own data and never shares it.  All the server ever
receives is a parameter *delta* — and even that can be sparsified
(top-k) or noised (differential privacy) before it leaves the client.
"""
from __future__ import annotations

import numpy as np

from .model import (accuracy, add_gaussian_noise, apply_delta, init_params,
                    param_delta, sparsify_topk, train_batch)


class Client:
    """One federated participant with private local data."""

    def __init__(self, client_id: int, X, y, X_test=None, y_test=None):
        self.client_id = int(client_id)
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.X_test = None if X_test is None else np.asarray(X_test, dtype=float)
        self.y_test = None if y_test is None else np.asarray(y_test)
        self.n = len(self.y)

    def local_train(self, global_params: dict, epochs: int = 2,
                    lr: float = 0.05, batch_size: int = 32, seed: int = 0,
                    prox_mu: float = 0.0, dp_sigma: float = 0.0,
                    topk: float = 1.0) -> dict:
        """Train locally from ``global_params`` and return the update.

        Returns a dict with the (possibly compressed/noised) ``delta``,
        the client's sample count ``n``, and local train accuracy before
        and after the local update.
        """
        before = accuracy(global_params, self.X, self.y)
        trained = train_batch(global_params, self.X, self.y, lr=lr,
                              epochs=epochs, batch_size=batch_size,
                              seed=seed, prox_params=global_params,
                              prox_mu=prox_mu)
        delta = param_delta(trained, global_params)
        if topk < 1.0:
            delta = sparsify_topk(delta, topk)
        if dp_sigma > 0.0:
            delta = add_gaussian_noise(delta, dp_sigma,
                                       seed=1000 + seed + self.client_id)
        after = accuracy(apply_delta(global_params, delta), self.X, self.y)
        return {"client_id": self.client_id, "delta": delta, "n": self.n,
                "acc_before": before, "acc_after": after}

    def local_accuracy(self, params: dict) -> float | None:
        """Accuracy of ``params`` on this client's held-out slice."""
        if self.X_test is None:
            return None
        return accuracy(params, self.X_test, self.y_test)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Client({self.client_id}, n={self.n})"


def train_local_only(X, y, epochs: int = 40, lr: float = 0.05,
                     seed: int = 0) -> dict:
    """Baseline: one client training alone, never seeing anyone else's data."""
    params = init_params(seed)
    return train_batch(params, X, y, lr=lr, epochs=epochs, seed=seed)
