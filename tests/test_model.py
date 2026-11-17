import numpy as np
import pytest

from fedlearn.data import generate_data
from fedlearn.model import (accuracy, add_gaussian_noise, apply_delta,
                            average_params, forward, init_params,
                            loss_and_grads, num_params, param_delta,
                            predict, sparsify_topk, train_batch)


def test_init_params_shapes():
    p = init_params(0)
    assert p["W1"].shape == (8, 24)
    assert p["b1"].shape == (24,)
    assert p["W2"].shape == (24, 4)
    assert p["b2"].shape == (4,)
    assert num_params(p) == 8 * 24 + 24 + 24 * 4 + 4


def test_forward_probabilities_sum_to_one():
    p = init_params(0)
    X, _, _, _ = generate_data(64, 16, seed=0)
    probs, _ = forward(p, X)
    assert probs.shape == (64, 4)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()


def test_loss_decreases_with_training():
    X, y, _, _ = generate_data(400, 40, seed=0)
    p = init_params(0)
    loss0, _ = loss_and_grads(p, X, y)
    trained = train_batch(p, X, y, lr=0.1, epochs=5, seed=0)
    loss1, _ = loss_and_grads(trained, X, y)
    assert loss1 < loss0
    assert accuracy(trained, X, y) > 0.75


def test_training_is_deterministic():
    X, y, _, _ = generate_data(200, 40, seed=0)
    a = train_batch(init_params(0), X, y, epochs=2, seed=1)
    b = train_batch(init_params(0), X, y, epochs=2, seed=1)
    assert np.allclose(a["W1"], b["W1"])


def test_grads_match_finite_differences():
    X, y, _, _ = generate_data(32, 8, seed=0)
    p = init_params(0)
    _, grads = loss_and_grads(p, X, y)
    eps = 1e-5
    for key, idx in [("W1", (0, 0)), ("W2", (0, 0)), ("b1", (0,)),
                     ("b2", (0,))]:
        orig = p[key][idx]
        p[key][idx] = orig + eps
        lp, _ = loss_and_grads(p, X, y)
        p[key][idx] = orig - eps
        lm, _ = loss_and_grads(p, X, y)
        p[key][idx] = orig
        numeric = (lp - lm) / (2 * eps)
        assert abs(numeric - grads[key][idx]) < 1e-4, (key, numeric,
                                                       grads[key][idx])


def test_param_delta_and_apply_roundtrip():
    a = init_params(0)
    b = init_params(1)
    assert np.allclose(apply_delta(a, param_delta(b, a))["W1"], b["W1"])


def test_average_params_weighted():
    a = init_params(0)
    b = init_params(1)
    avg = average_params([a, b], weights=[3, 1])
    assert np.allclose(avg["W1"], 0.75 * a["W1"] + 0.25 * b["W1"])
    with pytest.raises(ValueError):
        average_params([])


def test_sparsify_topk():
    delta = {"W": np.array([[0.1, -5.0], [0.2, 0.3]])}
    sparse = sparsify_topk(delta, 0.5)
    assert sparse["W"][0, 1] == -5.0
    assert np.count_nonzero(sparse["W"]) <= 2
    assert sparsify_topk(delta, 1.0)["W"].shape == (2, 2)
    with pytest.raises(ValueError):
        sparsify_topk(delta, 0.0)


def test_add_gaussian_noise():
    delta = {"W": np.zeros((4, 4))}
    assert np.allclose(add_gaussian_noise(delta, 0.0)["W"], 0.0)
    noisy = add_gaussian_noise(delta, 1.0, seed=0)
    assert not np.allclose(noisy["W"], 0.0)
    assert noisy["W"].shape == (4, 4)
    with pytest.raises(ValueError):
        add_gaussian_noise(delta, -1.0)


def test_predict_and_accuracy():
    X, y, _, _ = generate_data(200, 40, seed=0)
    p = train_batch(init_params(0), X, y, lr=0.1, epochs=5, seed=0)
    preds = predict(p, X)
    assert preds.shape == (200,)
    assert set(np.unique(preds)) <= set(range(4))
    assert accuracy(p, X, y) > 0.7
