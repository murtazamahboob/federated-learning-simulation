"""Federated learning simulation — no raw data ever leaves a client."""
from .simulate import compare, run_centralized, run_federated, run_local_only

__all__ = ["compare", "run_centralized", "run_federated", "run_local_only"]
__version__ = "1.0.0"
