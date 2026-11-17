"""End-to-end experiments: federated vs centralized vs local-only.

The headline comparison is the honest one for federated learning:
* centralised — all data in one place (the upper bound, and the
  privacy-violating option);
* local-only — each client trains alone (the privacy-preserving option
  with no collaboration);
* federated (FedAvg) — collaboration without sharing raw data.

A good federated run should sit well above local-only and close to
centralised — that gap is the whole value proposition.
"""
from __future__ import annotations

import numpy as np

from .client import Client, train_local_only
from .data import (N_CLASSES, generate_data, partition_iid, partition_noniid,
                   partition_skew)
from .model import accuracy, init_params, train_batch
from .server import Server


def build_clients(X_train, y_train, X_test, y_test, n_clients: int = 10,
                  seed: int = 0, noniid: bool = True):
    """Partition train and test the same way, so each client has a
    matching held-out slice for per-client (fairness) evaluation."""
    partition = partition_noniid if noniid else partition_iid
    train_parts = partition(X_train, y_train, n_clients, seed=seed)
    test_parts = partition(X_test, y_test, n_clients, seed=seed + 1)
    clients = []
    for i, ((X, y), (Xt, yt)) in enumerate(zip(train_parts, test_parts)):
        clients.append(Client(i, X, y, Xt, yt))
    return clients


def run_federated(seed: int = 0, rounds: int = 20, n_clients: int = 10,
                  fraction: float = 0.5, local_epochs: int = 2,
                  lr: float = 0.05, noniid: bool = True,
                  prox_mu: float = 0.0, dp_sigma: float = 0.0,
                  topk: float = 1.0, dropout_rate: float = 0.0,
                  n_train: int = 2000, n_test: int = 400,
                  noise: float = 1.8) -> dict:
    """Run a full federated experiment and return its summary."""
    X_train, y_train, X_test, y_test = generate_data(
        n_train, n_test, seed=seed, noise=noise)
    clients = build_clients(X_train, y_train, X_test, y_test,
                            n_clients=n_clients, seed=seed, noniid=noniid)
    server = Server(clients, seed=seed)
    initial_acc = accuracy(server.params, X_test, y_test)
    history = server.run(X_test, y_test, rounds=rounds, fraction=fraction,
                         local_epochs=local_epochs, lr=lr, prox_mu=prox_mu,
                         dp_sigma=dp_sigma, topk=topk,
                         dropout_rate=dropout_rate)
    final = history[-1] if history else {}
    return {
        "method": "federated",
        "seed": seed, "rounds": rounds, "n_clients": n_clients,
        "fraction": fraction, "local_epochs": local_epochs, "lr": lr,
        "noniid": noniid, "prox_mu": prox_mu, "dp_sigma": dp_sigma,
        "topk": topk, "dropout_rate": dropout_rate,
        "partition_skew": partition_skew(
            [(c.X, c.y) for c in clients]),
        "initial_test_accuracy": initial_acc,
        "final_test_accuracy": final.get("test_accuracy"),
        "mean_client_accuracy": final.get("mean_client_accuracy"),
        "min_client_accuracy": final.get("min_client_accuracy"),
        "max_client_accuracy": final.get("max_client_accuracy"),
        "bytes_sent_total": server.bytes_sent,
        "history": history,
    }


def run_centralized(seed: int = 0, epochs: int = 40, lr: float = 0.05,
                    n_train: int = 2000, n_test: int = 400,
                    noise: float = 1.8) -> dict:
    """All data pooled in one place — the accuracy upper bound."""
    X_train, y_train, X_test, y_test = generate_data(
        n_train, n_test, seed=seed, noise=noise)
    params = train_batch(init_params(seed), X_train, y_train,
                         lr=lr, epochs=epochs, seed=seed)
    return {"method": "centralized", "seed": seed, "epochs": epochs,
            "final_test_accuracy": accuracy(params, X_test, y_test)}


def run_local_only(seed: int = 0, epochs: int = 40, lr: float = 0.05,
                   n_clients: int = 10, noniid: bool = True,
                   n_train: int = 2000, n_test: int = 400,
                   noise: float = 1.8) -> dict:
    """Every client trains alone — the no-collaboration baseline."""
    X_train, y_train, X_test, y_test = generate_data(
        n_train, n_test, seed=seed, noise=noise)
    clients = build_clients(X_train, y_train, X_test, y_test,
                            n_clients=n_clients, seed=seed, noniid=noniid)
    accs = []
    for i, client in enumerate(clients):
        params = train_local_only(client.X, client.y, epochs=epochs,
                                  lr=lr, seed=seed + i)
        accs.append(accuracy(params, X_test, y_test))
    return {"method": "local-only", "seed": seed, "epochs": epochs,
            "n_clients": n_clients, "noniid": noniid,
            "final_test_accuracy": float(np.mean(accs)),
            "min_client_accuracy": float(np.min(accs)),
            "max_client_accuracy": float(np.max(accs)),
            "per_client_accuracy": [float(a) for a in accs]}


def compare(seed: int = 0, rounds: int = 20, n_clients: int = 10,
            **fed_kwargs) -> dict:
    """Federated vs centralised vs local-only, same seed and data.

    Centralised/local-only get ``rounds * local_epochs`` epochs so the
    total amount of gradient work is comparable.
    """
    local_epochs = fed_kwargs.get("local_epochs", 2)
    epochs = rounds * local_epochs
    fed = run_federated(seed=seed, rounds=rounds, n_clients=n_clients,
                        **fed_kwargs)
    central = run_centralized(seed=seed, epochs=epochs,
                              lr=fed_kwargs.get("lr", 0.05),
                              noise=fed_kwargs.get("noise", 1.8))
    local = run_local_only(seed=seed, epochs=epochs,
                           lr=fed_kwargs.get("lr", 0.05),
                           n_clients=n_clients,
                           noniid=fed_kwargs.get("noniid", True),
                           noise=fed_kwargs.get("noise", 1.8))
    return {"seed": seed, "rounds": rounds,
            "federated": fed, "centralized": central, "local_only": local}
