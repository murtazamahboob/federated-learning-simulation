"""Markdown + JSON reports for a federated experiment."""
from __future__ import annotations

import json
import os


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def write_report(result: dict, out_dir: str) -> tuple:
    """Write report.md and summary.json for a federated result (or a
    ``compare`` dict, in which case the federated part is reported and
    the baselines are tabulated). Returns (md_path, json_path)."""
    os.makedirs(out_dir, exist_ok=True)
    fed = result.get("federated", result)
    lines = [
        "# Federated Learning Simulation — Report",
        "",
        f"- Method: FedAvg, {fed['rounds']} rounds, "
        f"{fed['n_clients']} clients, {fed['fraction'] * 100:.0f}% sampled/round",
        f"- Local training: {fed['local_epochs']} epochs/round, lr {fed['lr']}",
        f"- Data: {'non-IID (label skew)' if fed['noniid'] else 'IID'}, "
        f"partition skew {fed['partition_skew']:.3f}",
        f"- FedProx mu: {fed['prox_mu']}, DP sigma: {fed['dp_sigma']}, "
        f"top-k keep: {fed['topk']}, dropout: {fed['dropout_rate']}",
        "",
        "## Results",
        "",
        f"- Test accuracy: {_pct(fed['initial_test_accuracy'])} "
        f"(round 0) → **{_pct(fed['final_test_accuracy'])}** "
        f"(round {fed['rounds']})",
        f"- Per-client accuracy: mean {_pct(fed['mean_client_accuracy'])}, "
        f"worst client {_pct(fed['min_client_accuracy'])}, "
        f"best {_pct(fed['max_client_accuracy'])}",
        f"- Communication: {fed['bytes_sent_total']:,} bytes "
        f"({fed['bytes_sent_total'] / 1024:.1f} KiB) client → server",
        "",
    ]
    if "centralized" in result and "local_only" in result:
        lines += [
            "## Baselines (same seed, same data)",
            "",
            "| Method | Test accuracy |",
            "| --- | --- |",
            f"| Centralised (all data pooled) | "
            f"{_pct(result['centralized']['final_test_accuracy'])} |",
            f"| Federated (FedAvg, no raw data shared) | "
            f"{_pct(fed['final_test_accuracy'])} |",
            f"| Local-only (no collaboration) | "
            f"{_pct(result['local_only']['final_test_accuracy'])} |",
            "",
            "Federated learning should sit well above local-only and "
            "close to centralised — collaboration without sharing data.",
            "",
        ]
    lines += [
        "## Accuracy by round",
        "",
        "| Round | Test accuracy | Mean client | Worst client | "
        "Bytes sent (total) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for h in fed["history"]:
        lines.append(
            f"| {h['round']} | {_pct(h['test_accuracy'])} | "
            f"{_pct(h['mean_client_accuracy'])} | "
            f"{_pct(h['min_client_accuracy'])} | "
            f"{h['bytes_sent_total']:,} |")
    lines.append("")
    md_path = os.path.join(out_dir, "report.md")
    json_path = os.path.join(out_dir, "summary.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    summary = {k: v for k, v in fed.items() if k != "history"}
    summary["history"] = fed["history"]
    if "centralized" in result:
        summary["baselines"] = {
            "centralized": result["centralized"]["final_test_accuracy"],
            "local_only": result["local_only"]["final_test_accuracy"],
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return md_path, json_path
