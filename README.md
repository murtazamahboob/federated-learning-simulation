# Federated Learning Simulation — FedAvg, FedProx, DP Noise & Top-k Compression in Pure NumPy

Train a neural network across 10 clients **without any raw data ever leaving a client**. Each client trains locally on its own non-IID, label-skewed data and shares only model updates with the server, which averages them (FedAvg). The simulation benchmarks federated learning against the two honest extremes — centralized training (all data pooled) and local-only training (no collaboration) — and shows exactly what privacy, compression and dropout cost.

**Headline result (seed 0, 20 rounds): federated 91.2% test accuracy — essentially matching centralized training at 91.5% — while local-only training reaches only 87.1%. No raw data shared.**

## Why this is interesting

- **Non-IID data, honestly simulated** — every client has two dominant classes (80% of its data) and only a little of the rest, the realistic federated case. An IID mode is included so you can measure exactly how much skew costs.
- **FedAvg + FedProx** — weighted averaging by client sample count, plus an optional proximal term that keeps clients from drifting too far from the global model on skewed data.
- **Privacy, measured** — add Gaussian noise to updates (differential-privacy style) and watch accuracy fall as privacy rises.
- **Communication, counted in bytes** — top-k sparsification sends only the largest update entries; the report shows the byte bill shrinking round by round.
- **Real-world messiness** — per-round client sampling and client dropout (a phone going offline mid-round simply contributes nothing).
- **Fairness, not just mean accuracy** — the report tracks the worst client's accuracy alongside the mean, so a model that works for 9 clients and fails the 10th can't hide.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Run the headline experiment: federated vs centralized vs local-only
fedlearn compare

# Plain FedAvg, 20 rounds, half the clients sampled per round
fedlearn simulate --rounds 20 --clients 10 --fraction 0.5

# Harder settings: FedProx + DP noise + top-k compression + 20% dropout
fedlearn simulate --prox-mu 0.1 --dp-sigma 0.05 --topk 0.25 --dropout 0.2 --out outputs

# Baselines on their own
fedlearn centralized
fedlearn local-only

# Full demo with report
python examples/demo.py
```

## How it works

```
synthetic data (4 classes, 8 features, Gaussian blobs)
        │
        ▼
non-IID partition ──► 10 clients, label skew (partition skew 0.300)
        │
        ▼  each round: sample 50% of clients
┌───────────────┐
│ client trains │  2 local epochs, small MLP (8→24→4) from scratch in NumPy
│ locally       │  optional FedProx term, DP noise, top-k sparsification
└───────┬───────┘
        │  only the parameter delta leaves the client
        ▼
server: FedAvg ──► weighted average of deltas (by client sample count)
        │
        ▼
evaluate: global test accuracy + per-client accuracy + bytes sent
```

The MLP, the data generator, the partitioner and the protocol are all built from scratch — NumPy is the only dependency. Gradients are verified against finite differences in the test suite.

## Results

| Method | Test accuracy |
| --- | --- |
| Centralized (all data pooled) | 91.5% |
| **Federated (FedAvg, no raw data shared)** | **91.2%** |
| Local-only (no collaboration) | 87.1% |

Federated learning starts at 21.2% (round 0) and reaches 91.2% by round 20; the worst client still scores 85.0%. Total communication: 252,800 bytes client → server over 20 rounds. See `outputs/report.md` for the per-round table.

## Project structure

```
src/fedlearn/
  data.py       synthetic data + non-IID / IID partitions, skew metric
  model.py      MLP from scratch (forward/backward/SGD), FedAvg averaging,
                top-k sparsification, Gaussian DP noise
  client.py     local training; only a parameter delta ever leaves a client
  server.py     FedAvg rounds, client sampling, dropout, byte accounting
  simulate.py   federated / centralized / local-only experiments + compare
  report.py     Markdown + JSON reports
  cli.py        `fedlearn simulate|compare|centralized|local-only`
examples/demo.py   headline experiment, writes outputs/report.md
tests/             40 tests (data, model, client, server, simulation, CLI)
```

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q   # 40 passed
```

## License

MIT — see [LICENSE](LICENSE).
