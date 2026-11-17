import numpy as np
import pytest

from fedlearn.data import (N_CLASSES, N_FEATURES, class_histogram,
                           generate_data, make_centres, partition_iid,
                           partition_noniid, partition_skew)


def test_generate_data_shapes_and_balance():
    Xtr, ytr, Xte, yte = generate_data(2000, 400, seed=0)
    assert Xtr.shape == (2000, N_FEATURES)
    assert Xte.shape == (400, N_FEATURES)
    assert set(np.unique(ytr)) == set(range(N_CLASSES))
    for c in range(N_CLASSES):
        assert (ytr == c).sum() == 500
        assert (yte == c).sum() == 100


def test_generate_data_deterministic():
    a = generate_data(200, 40, seed=3)
    b = generate_data(200, 40, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = generate_data(200, 40, seed=4)
    assert not np.array_equal(a[0], c[0])


def test_centres_separated():
    centres = make_centres(0)
    assert centres.shape == (N_CLASSES, N_FEATURES)
    for i in range(N_CLASSES):
        for j in range(i + 1, N_CLASSES):
            assert np.linalg.norm(centres[i] - centres[j]) > 2.0


def test_noniid_partition_covers_all_data_once():
    X, y, _, _ = generate_data(2000, 400, seed=0)
    clients = partition_noniid(X, y, n_clients=10, seed=0)
    assert len(clients) == 10
    total = sum(len(cy) for _, cy in clients)
    assert total == 2000
    # Every client has data, and (label skew) a dominant class pair.
    for _, cy in clients:
        assert len(cy) == 200
        hist = class_histogram(cy)
        assert max(hist.values()) >= 60  # dominant classes dominate
    # No sample is used twice: class counts across clients sum to the
    # global counts.
    total_hist = {c: 0 for c in range(N_CLASSES)}
    for _, cy in clients:
        for c, n_c in class_histogram(cy).items():
            total_hist[c] += n_c
    assert total_hist == class_histogram(y)


def test_noniid_more_skewed_than_iid():
    X, y, _, _ = generate_data(2000, 400, seed=0)
    noniid = partition_noniid(X, y, n_clients=10, seed=0)
    iid = partition_iid(X, y, n_clients=10, seed=0)
    assert partition_skew(noniid) > partition_skew(iid)
    assert partition_skew(iid) < 0.10
    assert partition_skew(noniid) > 0.15


def test_partition_errors():
    X, y, _, _ = generate_data(200, 40, seed=0)
    with pytest.raises(ValueError):
        partition_noniid(X, y, n_clients=0)
    with pytest.raises(ValueError):
        partition_iid(X, y, n_clients=0)
    with pytest.raises(ValueError):
        partition_noniid(X, y, dominant_share=1.5)


def test_class_histogram():
    assert class_histogram(np.array([0, 0, 1, 3])) == {0: 2, 1: 1, 2: 0, 3: 1}
