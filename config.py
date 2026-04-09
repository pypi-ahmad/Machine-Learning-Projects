"""
Shared workspace configuration — single source of truth for paths and constants.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
REGISTRY_PATH = ARTIFACTS / "global_registry.json"
LEADERBOARD_CSV = ARTIFACTS / "leaderboard.csv"
LEADERBOARD_PNG = ARTIFACTS / "leaderboard.png"
