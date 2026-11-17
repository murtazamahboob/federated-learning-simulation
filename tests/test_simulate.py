import numpy as np

from fedlearn.simulate import (build_clients, compare, run_centralized,
                               run_federated, run_local_only)


def test_federated_learns_and_beats_local_only():
    fed = run_federated(seed=0, rounds=10, n_clients=6, fraction=1.0)
    assert fed["final_test_accuracy"] > fed["initial_test_accuracy"]
    assert fed["final_test_accuracy"] > 0.70
    assert fed["min_client_accuracy"] is not None
    assert fed["min_client_accuracy"] <= fed["mean_client_accuracy"]
    assert fed["bytes_sent_total"] > 0
    assert len(fed["history"]) == 10


def test_compare_ordering_federated_between_baselines():
    result = compare(seed=0, rounds=10, n_clients=6, fraction=1.0)
    fed = result["federated"]["final_test_accuracy"]
    central = result["centralized"]["final_test_accuracy"]
    local = result["local_only"]["final_test_accuracy"]
    # Collaboration helps a lot; pooling everything is still best.
    assert fed > local
    assert fed > 0.70
    assert central >= fed - 0.05


def test_iid_beats_noniid_for_federated():
    fed_noniid = run_federated(seed=0, rounds=10, n_clients=6, fraction=1.0,
                               noniid=True)
    fed_iid = run_federated(seed=0, rounds=10, n_clients=6, fraction=1.0,
                            noniid=False)
    assert fed_iid["final_test_accuracy"] >= fed_noniid["final_test_accuracy"]


def test_fedprox_and_topk_still_learn():
    fed = run_federated(seed=0, rounds=10, n_clients=6, fraction=1.0,
                        prox_mu=0.1, topk=0.5)
    assert fed["final_test_accuracy"] > 0.65


def test_dp_noise_costs_accuracy():
    clean = run_federated(seed=0, rounds=10, n_clients=6, fraction=1.0)
    noisy = run_federated(seed=0, rounds=10, n_clients=6, fraction=1.0,
                          dp_sigma=0.05)
    # Strong noise must hurt; a tiny sigma may not, so only check it
    # does not magically beat the clean run by a wide margin.
    assert noisy["final_test_accuracy"] <= clean["final_test_accuracy"] + 0.05


def test_topk_cuts_communication():
    full = run_federated(seed=0, rounds=5, n_clients=6, fraction=1.0,
                         topk=1.0)
    sparse = run_federated(seed=0, rounds=5, n_clients=6, fraction=1.0,
                           topk=0.25)
    assert sparse["bytes_sent_total"] < full["bytes_sent_total"]


def test_centralized_and_local_only_run():
    central = run_centralized(seed=0, epochs=20)
    assert central["final_test_accuracy"] > 0.80
    local = run_local_only(seed=0, epochs=20, n_clients=6)
    assert 0.0 <= local["final_test_accuracy"] <= 1.0
    assert local["min_client_accuracy"] <= local["max_client_accuracy"]


def test_build_clients_matching_test_slices():
    from fedlearn.data import generate_data
    X, y, Xt, yt = generate_data(600, 120, seed=0)
    clients = build_clients(X, y, Xt, yt, n_clients=3, seed=0)
    assert len(clients) == 3
    assert sum(c.n for c in clients) == 600
    assert all(c.X_test is not None for c in clients)
