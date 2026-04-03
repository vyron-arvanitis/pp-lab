from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLUMNS_BY_COORDINATE_SYSTEM = {
    "cartesian": ["prodTime", "x", "y", "z", "energy", "px", "py", "pz"],
    "cylindrical": ["r", "z", "p_xy", "pz", "prodTime", "energy"],
}

DEFAULT_DATASET_PATH = PROJECT_ROOT / "smartbkg_dataset_4k.parquet"
DEFAULT_TRAINING_DATASET_PATH = PROJECT_ROOT / "smartbkg_dataset_4k_training.parquet"
PDG_MAPPING_PATH = PROJECT_ROOT / "pdg_mapping.json"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
FULL_TRAIN_MODELS_DIR = PROJECT_ROOT / "models_fulltrain"
OPTUNA_RESULTS_DIR = PROJECT_ROOT / "optuna_results"
EMBEDDING_ANALYSIS_DIR = PROJECT_ROOT / "pdg_embedding_analysis"
MODEL_COMPARISON_DIR = PROJECT_ROOT / "compare_model_features"


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def ensure_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def get_feature_columns(coordinates: str) -> list[str]:
    try:
        return list(FEATURE_COLUMNS_BY_COORDINATE_SYSTEM[coordinates])
    except KeyError as exc:
        valid = ", ".join(sorted(FEATURE_COLUMNS_BY_COORDINATE_SYSTEM))
        raise ValueError(f"Unknown coordinate system '{coordinates}'. Expected one of: {valid}.") from exc


def load_pdg_mapping(path: str | Path = PDG_MAPPING_PATH) -> dict[int, int]:
    with resolve_path(path).open(encoding="utf-8") as handle:
        return dict(json.load(handle))
