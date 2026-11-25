"""A stand-in for the inference service, for tests only (the app itself has no offline mode).

Same interface as firstlook.backend.RemoteBackend; patients are invented, not ADNI records."""
from __future__ import annotations

import copy
import io

from firstlook.backend import BackendError, Patient, Prediction


def _png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (60, 20), "black").save(buf, "PNG")
    return buf.getvalue()


SHAP = {"APOE4": 0.0213, "FAQ": -0.0145, "Sex": -0.0053, "Age": -0.0011, "MMSE": -0.0022, "GDS": -0.0014}
BASE = {"age": 72, "sex": "F", "apoe": "ε3/ε4", "mmse": 26, "faq": 3, "gds": 2, "npiq": 1, "weight": 64}

# image ID -> (record changes, missing, mri available, five fold probabilities, held-out fold)
PATIENTS = {
    "P-HIGH": ({"mmse": 19, "faq": 21}, [], True, [0.80, 0.84, 0.78, 0.86, 0.82], 2),
    "P-LOW": ({"mmse": 29, "faq": 0}, [], True, [0.12, 0.15, 0.10, 0.18, 0.14], 4),
    "P-MID": ({}, [], True, [0.48, 0.45, 0.44, 0.52, 0.49], 2),
    "P-SPLIT": ({"gds": 12}, [], True, [0.20, 0.81, 0.47, 0.66, 0.30], 3),
    "P-NOMRI": ({}, [], False, [0.20, 0.22, 0.19, 0.21, 0.20], 1),
    "P-OLD": ({"age": 95}, [], True, [0.30, 0.28, 0.33, 0.31, 0.29], 5),
    "P-MISS": ({"mmse": None, "gds": None}, ["mmse", "gds"], True, [0.45, 0.43, 0.48, 0.52, 0.44], 2),
}


class FakeBackend:
    name = "inference service"

    def __init__(self, patients=None):
        self.patients = copy.deepcopy(patients or PATIENTS)

    def health(self):
        return {"ok": True, "patients": len(self.patients), "device": "test"}

    def _p(self, image_id):
        if image_id not in self.patients:
            raise KeyError(image_id)
        return self.patients[image_id]

    def patient(self, image_id):
        changes, missing, mri, _, _ = self._p(image_id)
        values = {**BASE, **changes}
        return Patient(image_id, values, list(missing), mri, "service")

    def mri_png(self, image_id):
        self._p(image_id)
        return _png()

    def attribution_png(self, image_id):
        self._p(image_id)
        return _png()

    def upload_mri(self, image_id, filename, data):
        raise BackendError("uploads are not part of the fake service")

    def predict(self, image_id):
        _, _, mri, probs, held = self._p(image_id)
        folds = [{"fold": i, "prob": p, "held_out": i == held} for i, p in enumerate(probs, 1)]
        return Prediction(mri, folds, held, probs[held - 1], round(sum(probs) / 5, 4),
                          round(max(probs) - min(probs), 4), dict(SHAP), 0.45, "service")
