"""Every path in the project, defined in one place.

Modules import from here instead of computing ``Path(__file__).parents[N]``
themselves.  Counting directory levels breaks the moment a file is moved to a
different folder, and it breaks silently -- the code still runs, it just writes
its output somewhere nobody looks.

Usage:

    from src.paths import SE_DIR, FIGURES
    df = pd.read_csv(SE_DIR / "clean.csv", sep=";")
"""

from pathlib import Path

# src/paths.py -> src/ -> project root
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- input data
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_SENTI4SD = RAW / "senti4sd"
RAW_DRUGLIB = RAW / "druglib"

PROCESSED = DATA / "processed"
SE_DIR = PROCESSED / "se"
HEALTH_DIR = PROCESSED / "health"

STRESS_TEST = DATA / "stress_test"
LEXICONS = DATA / "lexicons"

# --------------------------------------------------------------- output data
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
METRICS_TEST = RESULTS / "metrics_test"
FIGURES = RESULTS / "figures"
ABLATION = RESULTS / "ablation"
FEATURES = RESULTS / "features"
EXPLANATIONS = RESULTS / "explanations"

MODELS = ROOT / "models"
REPORT = ROOT / "report"


def processed_dir(domain: str) -> Path:
    """Return the processed-data folder for ``"se"`` or ``"health"``."""
    if domain == "se":
        return SE_DIR
    if domain == "health":
        return HEALTH_DIR
    raise ValueError(f"domain must be 'se' or 'health', got {domain!r}")
