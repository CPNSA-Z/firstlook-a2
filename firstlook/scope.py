"""Deterministic input checks and model-uncertainty checks.

Two separate rule sets, so that no model output can reach Step 1:
  * input_checks(record)  - look only at the record; shown in Step 1, before the clinician's
    own judgement. They cannot leak a score because they never receive one.
  * model_checks(view)    - look only at the five fold scores; shown in Step 3, next to the
    fold range, after the clinician's judgement is locked.
Neither uses the LLM, so safety-relevant alerts never depend on generated text.
Each alert names the field it comes from (provenance) and says what the reviewer can do.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import TRAINING_RANGES
from .model_adapter import EnsembleView

FIELD_NAMES = {"age": "age", "mmse": "MMSE", "faq": "FAQ", "gds": "GDS"}
GDS_DEPRESSION_CUTOFF = 10   # GDS-30: 10+ suggests clinically relevant depressive symptoms


@dataclass
class Alert:
    level: str        # "block" = model output should not be used; "caution" = use with care
    field: str        # provenance: which record field or model output triggered it
    message: str
    action: str


def input_checks(record: dict) -> list[Alert]:
    """Step 1 rules. Receives the record only - never a model score."""
    alerts: list[Alert] = []

    status = str(record.get("mri_status", "")).lower()
    if "fail" in status or "zero" in status or not status:
        alerts.append(Alert(
            "block", "mri_status",
            "The MRI did not load. The pipeline substitutes a blank volume, so the model "
            "score will be withheld in Step 3.",
            "Judge from the history; re-run the model after the scan is available."))

    # Missing inputs. Numeric ones were median-imputed for the model; APOE and sex were silently coded
    # as "no e4" / female when absent. Either way part of the score rests on an assumed value.
    missing = list(record.get("_missing", [])) or [f for f in TRAINING_RANGES if record.get(f) is None]
    defaulted = list(record.get("_defaulted", []))
    if missing or defaulted:
        parts = []
        if missing:
            parts.append(", ".join(FIELD_NAMES.get(f, f) for f in missing)
                         + " not recorded, so the model used the cohort median")
        if "apoe" in defaulted:
            parts.append("APOE not recorded: the model treated the patient as a non-carrier (no ε4)")
        if "sex" in defaulted:
            parts.append("sex not recorded: the model treated the patient as female")
        alerts.append(Alert("caution", "missing", "Part of the model's input is assumed: " + "; ".join(parts) + ".",
                            "Weigh the AI result with care, or complete the record first."))

    for fid, (lo, hi) in TRAINING_RANGES.items():
        v = record.get(fid)
        if v is None:
            continue
        if not (lo <= float(v) <= hi):
            alerts.append(Alert(
                "caution", fid,
                f"{fid} = {v} is outside the range the model was trained on ({lo}-{hi}).",
                "The model has little evidence for this patient group; weigh history more."))

    gds = record.get("gds")
    if gds is not None and float(gds) >= GDS_DEPRESSION_CUTOFF:
        alerts.append(Alert(
            "caution", "gds",
            f"Depression scale is high (GDS = {gds}). Low mood can lower cognitive scores. "
            "The model receives the GDS number but not the reason for the low mood.",
            "Consider mood assessment before interpreting cognitive decline."))
    return alerts


def model_checks(view: EnsembleView) -> list[Alert]:
    """Step 3 rules. Shown only after the clinician's own judgement is locked."""
    alerts: list[Alert] = []
    if view.disagree:
        alerts.append(Alert(
            "caution", "model:fold_probs",
            f"The five models disagree (range {view.low:.2f}-{view.high:.2f}).",
            "Do not rely on the average; the evidence is not decisive for the models either."))
    elif view.borderline:
        alerts.append(Alert(
            "caution", "model:fold_probs",
            f"The estimate ({view.mean:.2f}) is near the decision line.",
            "Small changes in input could change the result."))
    return alerts


def model_usable(alerts: list[Alert]) -> bool:
    return not any(a.level == "block" for a in alerts)
