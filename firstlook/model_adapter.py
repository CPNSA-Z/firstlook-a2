"""Model adapter: turns the five fold outputs from the inference service into what the interface may show.

Design choice: the interface never receives a single verdict. It receives the spread of
five independently trained fold models, so disagreement is visible as uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass

DISAGREEMENT_SPREAD = 0.25   # max-min across folds at or above this = "models disagree"
BORDER_LOW, BORDER_HIGH = 0.35, 0.65   # mean inside this band = "near the decision line"


@dataclass
class EnsembleView:
    fold_probs: list[float]
    mean: float
    low: float
    high: float
    spread: float
    disagree: bool
    borderline: bool
    band: str          # "lower", "uncertain", "higher" - deliberately coarse wording
    attributions: dict[str, float]
    source: str = "service"


def view_from_prediction(pred) -> EnsembleView:
    """EnsembleView from a backend.Prediction. The held-out fold (which did not train on this
    patient) is the main estimate; the five folds give the spread."""
    probs = [float(p) for p in pred.probs]
    m, lo, hi = float(pred.held_out_prob), min(probs), max(probs)
    disagree = hi - lo >= DISAGREEMENT_SPREAD
    borderline = BORDER_LOW <= m <= BORDER_HIGH
    band = "uncertain" if (disagree or borderline) else ("higher" if m > BORDER_HIGH else "lower")
    return EnsembleView(probs, m, lo, hi, hi - lo, disagree, borderline, band, dict(pred.shap),
                        source=pred.source)
