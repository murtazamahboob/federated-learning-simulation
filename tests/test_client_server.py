import numpy as np
import pytest

from fedlearn.client import Client, train_local_only
from fedlearn.data import generate_data, partition_noniid
from fedlearn.model import accuracy, init_params
from fedlearn.server import Server, communication_bytes


def _clients(seed=0, n_clients=4):
    X, y, Xt, yt = generate_data(800, 200, seed=seed)
    parts = partition_noniid(X, y, n_clients=n_clients, seed=seed)
    tparts = partition_noniid(Xt, yt, n_clients=n_clients, seed=seed + 1)
    return [Client(i, Xp, yp, Xt_p, yt_p)
            for i, ((Xp, yp), (Xt_p, yt_p)) in enumerate(zip(parts, tparts))]


def test_client_local_train_improves_local_accuracy():
    clients = _clients()
    params = init_params(0)
    result = clients[0].local_train(params, epochs=3, seed=0)
    assert result["acc_after"] >= result["acc_before"]
    assert result["n"] == 200
    assert result["client_id"] == 0
    assert set(result["delta"]) == {"W1", "b1", "W2", "b2"}


def test_client_never_shares_raw_data():
    # The update contains only parameter-shaped arrays, never the data.
    clients = _clients()
    result = clients[0].local_train(init_params(0), epochs=1, seed=0)
    assert "X" not in result and "y" not in result
    assert result["delta"]["W1"].shape == (8, 24)


def test_client_topk_reduces_communication():
    clients = _clients()
    full = clients[0].local_train(init_params(0), epochs=1, seed=0)
    sparse = clients[0].local_train(init_params(0), epochs=1, seed=0,
                                    topk=0.1)
    assert (communication_bytes(sparse["delta"])
            < communication_bytes(full["delta"]))


def test_client_dp_noise_changes_update():
    clients = _clients()
    plain = clients[0].local_train(init_params(0), epochs=1, seed=0)
    noisy = clients[0].local_train(init_params(0), epochs=1, seed=0,
                                   dp_sigma=0.1)
    assert not np.allclose(plain["delta"]["W1"], noisy["delta"]["W1"])


def test_train_local_only_learns():
    X, y, Xt, yt = generate_data(800, 200, seed=0)
    params = train_local_only(X, y, epochs=10, seed=0)
    assert accuracy(params, Xt, yt) > 0.7


def test_communication_bytes_counts_nonzero():
    delta = {"W": np.array([[1.0, 0.0], [0.0, 2.0]]), "b": np.zeros(3)}
    assert communication_bytes(delta) == 2 * 8


def test_server_round_improves_test_accuracy():
    clients = _clients()
    X, y, Xt, yt = generate_data(800, 200, seed=0)
    server = Server(clients, seed=0)
    before = accuracy(server.params, Xt, yt)
    entry = server.run_round(Xt, yt, fraction=1.0, local_epochs=2)
    assert entry["round"] == 1
    assert entry["clients_sampled"] == 4
    assert entry["test_accuracy"] > before
    assert entry["bytes_sent_total"] > 0


def test_server_sampling_fraction():
    clients = _clients()
    server = Server(clients, seed=0)
    assert len(server.sample_clients(0.5)) == 2
    assert len(server.sample_clients(1.0)) == 4
    with pytest.raises(ValueError):
        server.sample_clients(0.0)


def test_server_dropout_drops_clients():
    clients = _clients()
    X, y, Xt, yt = generate_data(800, 200, seed=0)
    server = Server(clients, seed=0)
    entry = server.run_round(Xt, yt, fraction=1.0, dropout_rate=1.0)
    assert entry["clients_dropped"] == 4
    assert entry["bytes_sent_total"] == 0


def test_server_history_tracks_rounds():
    clients = _clients()
    _, _, Xt, yt = generate_data(800, 200, seed=0)
    server = Server(clients, seed=0)
    history = server.run(Xt, yt, rounds=3, fraction=0.5)
    assert len(history) == 3
    assert [h["round"] for h in history] == [1, 2, 3]
    assert history[-1]["bytes_sent_total"] >= history[0]["bytes_sent_total"]
