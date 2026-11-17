"""The server side: FedAvg rounds, client sampling, dropout, and the
communication bill.

Each round the server samples a fraction of clients, collects their
deltas, and averages them weighted by each client's sample count
(FedAvg).  Clients that drop out mid-round simply contribute nothing.
Communication is counted in bytes: a delta costs 8 bytes per non-zero
float64 entry (top-k sparsification therefore shows up directly as a
smaller bill).
"""
from __future__ import annotations

import numpy as np

from .model import accuracy, apply_delta, average_params, init_params

BYTES_PER_PARAM = 8


def communication_bytes(delta: dict) -> int:
    """Bytes to send one delta: 8 per non-zero entry."""
    return int(sum(int(np.count_nonzero(v)) for v in delta.values())
               * BYTES_PER_PARAM)


class Server:
    def __init__(self, clients, seed: int = 0):
        self.clients = list(clients)
        self.rng = np.random.default_rng(seed)
        self.params = init_params(seed)
        self.round = 0
        self.history = []          # one entry per round
        self.bytes_sent = 0        # total client -> server bytes

    def sample_clients(self, fraction: float) -> list:
        """Random subset of clients for one round."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        k = max(1, int(round(fraction * len(self.clients))))
        idx = self.rng.choice(len(self.clients), size=k, replace=False)
        return [self.clients[int(i)] for i in idx]

    def run_round(self, X_test, y_test, fraction: float = 0.5,
                  local_epochs: int = 2, lr: float = 0.05,
                  prox_mu: float = 0.0, dp_sigma: float = 0.0,
                  topk: float = 1.0, dropout_rate: float = 0.0) -> dict:
        """One federated round; returns the round's recorded metrics."""
        sampled = self.sample_clients(fraction)
        updates, weights, dropped = [], [], []
        for client in sampled:
            if dropout_rate > 0.0 and self.rng.random() < dropout_rate:
                dropped.append(client.client_id)
                continue
            result = client.local_train(
                self.params, epochs=local_epochs, lr=lr,
                seed=self.rng.integers(0, 1_000_000),
                prox_mu=prox_mu, dp_sigma=dp_sigma, topk=topk)
            self.bytes_sent += communication_bytes(result["delta"])
            updates.append(result["delta"])
            weights.append(result["n"])
        if updates:
            avg_delta = average_params(updates, weights)
            # average_params averages the deltas themselves, so apply it
            # as a weighted mean delta (weights are client sample counts).
            self.params = apply_delta(self.params, avg_delta)
        self.round += 1
        per_client = [c.local_accuracy(self.params) for c in self.clients]
        per_client = [a for a in per_client if a is not None]
        entry = {
            "round": self.round,
            "test_accuracy": accuracy(self.params, X_test, y_test),
            "clients_sampled": len(sampled),
            "clients_dropped": len(dropped),
            "dropped_ids": dropped,
            "mean_client_accuracy": (float(np.mean(per_client))
                                     if per_client else None),
            "min_client_accuracy": (float(np.min(per_client))
                                    if per_client else None),
            "max_client_accuracy": (float(np.max(per_client))
                                    if per_client else None),
            "bytes_sent_total": self.bytes_sent,
        }
        self.history.append(entry)
        return entry

    def run(self, X_test, y_test, rounds: int = 20, **kwargs) -> list:
        """Run ``rounds`` rounds; returns the full history."""
        for _ in range(rounds):
            self.run_round(X_test, y_test, **kwargs)
        return self.history
