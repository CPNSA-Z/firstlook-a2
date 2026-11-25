"""Shared constants."""
from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Approximate ranges seen in the 990-subject training cohort (cohort summary table; ADNI enrolment ages).
# Values outside these are "outside what the model has seen".
TRAINING_RANGES = {
    "age": (55, 90),
    "mmse": (15, 30),
}
