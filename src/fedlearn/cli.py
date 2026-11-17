"""CLI: fedlearn simulate|compare|centralized|local-only"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .report import write_report
from .simulate import compare, run_centralized, run_federated, run_local_only


def _add_common(p):
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--iid", action="store_true",
                   help="Use an IID partition instead of non-IID label skew")
    p.add_argument("--prox-mu", type=float, default=0.0,
                   help="FedProx proximal term (0 = plain FedAvg)")
    p.add_argument("--dp-sigma", type=float, default=0.0,
                   help="Gaussian noise std on updates (0 = no DP noise)")
    p.add_argument("--topk", type=float, default=1.0,
                   help="Fraction of largest update entries to send")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Per-round client dropout probability")
    p.add_argument("--out", default=None,
                   help="Directory for report.md / summary.json")


def _fed_kwargs(args):
    return dict(seed=args.seed, rounds=args.rounds, n_clients=args.clients,
                fraction=args.fraction, local_epochs=args.local_epochs,
                lr=args.lr, noniid=not args.iid, prox_mu=args.prox_mu,
                dp_sigma=args.dp_sigma, topk=args.topk,
                dropout_rate=args.dropout)


def _print_result(result):
    fed = result.get("federated", result)
    print(f"test accuracy: {100 * fed['initial_test_accuracy']:.1f}% "
          f"-> {100 * fed['final_test_accuracy']:.1f}% "
          f"over {fed['rounds']} rounds")
    print(f"worst client: {100 * fed['min_client_accuracy']:.1f}%  "
          f"communication: {fed['bytes_sent_total']:,} bytes")
    if "centralized" in result:
        print(f"centralized: "
              f"{100 * result['centralized']['final_test_accuracy']:.1f}%  "
              f"local-only: "
              f"{100 * result['local_only']['final_test_accuracy']:.1f}%")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="fedlearn",
        description="Simulate federated learning: clients train locally, "
                    "only model updates are shared.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="Run a federated (FedAvg) experiment")
    _add_common(s)
    c = sub.add_parser("compare",
                       help="Federated vs centralized vs local-only")
    _add_common(c)
    ce = sub.add_parser("centralized", help="Centralized baseline only")
    ce.add_argument("--seed", type=int, default=0)
    ce.add_argument("--epochs", type=int, default=40)
    ce.add_argument("--lr", type=float, default=0.05)
    lo = sub.add_parser("local-only", help="Local-only baseline only")
    lo.add_argument("--seed", type=int, default=0)
    lo.add_argument("--epochs", type=int, default=40)
    lo.add_argument("--lr", type=float, default=0.05)
    lo.add_argument("--clients", type=int, default=10)
    lo.add_argument("--iid", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "simulate":
        result = run_federated(**_fed_kwargs(args))
    elif args.cmd == "compare":
        result = compare(**_fed_kwargs(args))
    elif args.cmd == "centralized":
        result = run_centralized(seed=args.seed, epochs=args.epochs,
                                 lr=args.lr)
        print(f"centralized test accuracy: "
              f"{100 * result['final_test_accuracy']:.1f}%")
        return 0
    elif args.cmd == "local-only":
        result = run_local_only(seed=args.seed, epochs=args.epochs,
                                lr=args.lr, n_clients=args.clients,
                                noniid=not args.iid)
        print(f"local-only mean test accuracy: "
              f"{100 * result['final_test_accuracy']:.1f}% "
              f"(worst client {100 * result['min_client_accuracy']:.1f}%)")
        return 0
    else:  # pragma: no cover - argparse enforces cmd
        p.print_help()
        return 2

    _print_result(result)
    out = getattr(args, "out", None)
    if out:
        md, js = write_report(result, out)
        print(f"report: {md}")
        print(f"summary: {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
