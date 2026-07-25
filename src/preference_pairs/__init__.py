"""Build conversational DPO preference pairs from reflective-question traces."""

from .builder import build_preference_pairs
from .config import PreferencePairConfig, SourceConfig, load_preference_pair_config

__all__ = [
    "PreferencePairConfig",
    "SourceConfig",
    "build_preference_pairs",
    "load_preference_pair_config",
]
