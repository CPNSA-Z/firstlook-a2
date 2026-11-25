"""Where patient data and model results come from.

RemoteBackend - the FirstLook inference service on the GPU PC (real ADNI cohort, real models).
There is no offline mode: without the service the app says so and stops.

The app only ever passes an image ID; names are never stored or shown.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

SERVER_URL = os.getenv("FIRSTLOOK_SERVER_URL", "").rstrip("/")
SERVER_TOKEN = os.getenv("FIRSTLOOK_SERVER_TOKEN", "")

# display label, unit/notes; order = how the table is shown
FIELDS = {
    "age": "Age (years)",
    "sex": "Sex",
    "apoe": "APOE genotype",
    "mmse": "MMSE (0-30)",
    "faq": "FAQ functional score (0-30, higher = worse)",
    "gds": "Geriatric Depression Scale (GDS)",
    "npiq": "NPI-Q total",
    "weight": "Weight (kg)",
}
# The model's clinical inputs, verified against the ADNI export (exact match for all 990 scans):
# Sex, APOE4 carrier, Age, MMSE, Global CDR, FAQ, GDS. Global CDR is disabled: not in the database,
# not shown, replaced by the cohort mean for the model. NPI-Q and weight are shown but not model inputs.
MODEL_INPUTS = {"sex", "apoe", "age", "mmse", "faq", "gds"}
SHAP_LABEL = {"Sex": "Sex", "APOE4": "APOE4 carrier", "Age": "Age", "MMSE": "MMSE",
              "FAQ": "FAQ functional score", "GDS": "Depression scale (GDS)", "MRI image": "MRI image"}


@dataclass
class Patient:
    image_id: str
    values: dict                      # field id -> value (None if missing)
    missing: list = field(default_factory=list)
    mri_available: bool = True
    source: str = "service"
    mri_source: str = "stored"        # "stored" or "upload"
    defaulted: list = field(default_factory=list)   # no record, model assumed a default (apoe, sex)


@dataclass
class Prediction:
    mri_ok: bool
    folds: list                       # [{"fold", "prob", "held_out"}]
    held_out_fold: int | None
    held_out_prob: float
    mean: float
    spread: float
    shap: dict                        # input name -> contribution (probability units)
    shap_base: float | None
    source: str = "service"
    seconds: float | None = None
    estimate_mode: str = "held_out"   # "held_out" (stored scan) or "ensemble" (uploaded scan)

    @property
    def probs(self):
        return [f["prob"] for f in self.folds]


def _fmt(v):
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def patient_text(patient: "Patient") -> str:
    """The record as labelled lines: what the local LLM is allowed to use (also used for cache files)."""
    out = []
    for fid, label in FIELDS.items():
        if fid not in patient.values and fid not in patient.missing:
            continue
        v = _fmt(patient.values.get(fid))
        if fid in MODEL_INPUTS:
            out.append(f"{label}: {v if v is not None else 'not recorded (filled in for the model)'}")
        else:   # shown to the doctor, never seen by the model: say so, so the LLM cannot claim otherwise
            out.append(f"{label}: {v if v is not None else 'not recorded'} (not a model input)")
    out.append(f"T1 MRI: {'available' if patient.mri_available else 'NOT available'}")
    return "\n".join(out)


class BackendError(RuntimeError):
    pass


# ---------------------------------------------------------------- remote (inference service)
class RemoteBackend:
    name = "inference service"

    def __init__(self, url: str = SERVER_URL, token: str = SERVER_TOKEN, timeout: float = 120):
        import requests
        self.url, self.timeout = url.rstrip("/"), timeout
        self.s = requests.Session()
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"

    def _get(self, path, **params):
        try:
            r = self.s.get(self.url + path, params=params, timeout=self.timeout)
        except Exception as exc:
            raise BackendError(f"cannot reach {self.url}: {type(exc).__name__}") from exc
        if r.status_code == 404:
            raise KeyError(path.rsplit("/", 1)[-1])
        if r.status_code != 200:
            raise BackendError(f"{path}: HTTP {r.status_code} {r.text[:120]}")
        return r

    def health(self):
        return self._get("/health").json()

    def ids(self, prefix="", limit=20):
        return self._get("/ids", prefix=prefix, limit=limit).json()

    def patient(self, image_id):
        d = self._get(f"/patient/{image_id}").json()
        values = {k: d.get(k) for k in FIELDS if k in d}
        if "apoe_genotype" in d or "apoe4" in d:
            gt, carrier = d.get("apoe_genotype"), d.get("apoe4")
            values["apoe"] = gt.replace("e", "ε") if gt else (
                None if carrier is None else ("ε4 carrier" if carrier else "no ε4"))
        missing = [k for k in d.get("missing", []) if k in FIELDS]
        pt = Patient(d["image_id"], values, missing, d.get("mri_available", True), "service")
        pt.mri_source = d.get("mri_source", "stored")
        pt.defaulted = [k for k in d.get("defaulted", []) if k in FIELDS]
        return pt

    def mri_png(self, image_id):
        return self._get(f"/mri/{image_id}.png").content

    def upload_mri(self, image_id, filename, data: bytes):
        """Send a clinician-supplied scan; it replaces the stored one for this patient."""
        try:
            r = self.s.post(f"{self.url}/upload/{image_id}", params={"name": filename}, data=data,
                            headers={"Content-Type": "application/octet-stream"}, timeout=self.timeout)
        except Exception as exc:
            raise BackendError(f"cannot reach {self.url}: {type(exc).__name__}") from exc
        if r.status_code == 404:
            raise KeyError(image_id)
        if r.status_code != 200:
            raise BackendError(r.json().get("detail", r.text[:120]) if r.headers.get("content-type", "").startswith(
                "application/json") else f"HTTP {r.status_code}")
        return r.json()

    def predict(self, image_id):
        d = self._get(f"/predict/{image_id}").json()
        return Prediction(d["mri_ok"], d["folds"], d["held_out_fold"], d["held_out_prob"], d["mean"],
                          d["spread"], d["shap"], d.get("shap_base"), "service", d.get("seconds"),
                          d.get("estimate_mode", "held_out"))

    def attribution_png(self, image_id):
        return self._get(f"/attribution/{image_id}.png").content


def get_backend():
    """The inference service set in FIRSTLOOK_SERVER_URL. Raises BackendError if it is not set or not answering."""
    if not SERVER_URL:
        raise BackendError("FIRSTLOOK_SERVER_URL is not set (start the app with run_firstlook_lan.ps1).")
    b = RemoteBackend()
    b.health()
    return b
