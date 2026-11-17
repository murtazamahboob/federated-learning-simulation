"""Demo: federated vs centralized vs local-only, on non-IID data.

Runs the default 20-round FedAvg experiment plus both baselines and
writes outputs/report.md and outputs/summary.json.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedlearn.report import write_report
from fedlearn.simulate import compare


def main() -> int:
    result = compare(seed=0, rounds=20, n_clients=10)
    fed = result["federated"]
    print("Federated learning simulation (non-IID label skew, 10 clients)")
    print(f"  partition skew: {fed['partition_skew']:.3f}")
    print(f"  federated:   {100 * fed['final_test_accuracy']:.1f}% "
          f"(round 0: {100 * fed['initial_test_accuracy']:.1f}%)")
    print(f"  centralized: "
          f"{100 * result['centralized']['final_test_accuracy']:.1f}%")
    print(f"  local-only:  "
          f"{100 * result['local_only']['final_test_accuracy']:.1f}%")
    print(f"  worst client: {100 * fed['min_client_accuracy']:.1f}%  "
          f"communication: {fed['bytes_sent_total']:,} bytes")
    out = os.path.join(os.path.dirname(__file__), "..", "outputs")
    md, js = write_report(result, out)
    print(f"  report: {md}")
    print(f"  summary: {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
