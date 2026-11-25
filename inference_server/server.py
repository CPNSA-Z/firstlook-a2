"""FirstLook inference service. Runs on the GPU PC next to LM Studio.

    python server.py            (settings below, or environment variables)

It holds the patient table (built from df_cohort_final.pkl into a local SQLite file), the
preprocessed MRI volumes and the five fold checkpoints. The app only ever sends an image ID.
Nothing in this folder contains ADNI data; the data stays where FL_DATA_DIR points.

Endpoints (all JSON unless noted; send "Authorization: Bearer <FL_TOKEN>" if FL_TOKEN is set):
  GET /health                  models, device, number of patients
  GET /ids?prefix=I3&limit=20  image IDs (for the lookup box)
  GET /patient/{image_id}      basic information from the database (no outcome label)
  GET /mri/{image_id}.png      three orthogonal slices of the patient's T1 scan
  GET /predict/{image_id}      five fold scores, held-out fold, exact clinical Shapley values
  GET /attribution/{image_id}.png   smoothed Grad-CAM of the held-out fold model
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from sklearn.model_selection import StratifiedKFold

from explain import clinical_shapley, render_png, smooth_gradcam
from fl_model import CLINICAL_NAMES, TARGET_SHAPE, load_fold, preprocess_mri

DATA_DIR = Path(os.getenv("FL_DATA_DIR", "Clean_Data"))          # df_cohort_final.pkl + sub-*/
MODELS_DIR = Path(os.getenv("FL_MODELS_DIR", "models"))          # Twostage_PRO_fold{1..5}_*.pth
DB_PATH = Path(os.getenv("FL_DB", str(DATA_DIR / "firstlook_patients.db")))
META_CSV = Path(os.getenv("FL_META", str(DATA_DIR / "idaSearch_10_06_2025.csv")))  # ADNI export
TOKEN = os.getenv("FL_TOKEN", "")
DEVICE = os.getenv("FL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

# Global CDR is part of the diagnostic definition (leakage), so it is DISABLED everywhere: it is not
# stored in the patient table, never shown to the clinician, and the model receives the cohort mean
# (0 in z-space) in its place. Its Shapley value is then exactly 0 and it is left out of the output.
CDR = CLINICAL_NAMES.index("Global CDR")
SERVED_INPUTS = [n for n in CLINICAL_NAMES if n != "Global CDR"]
UPLOAD_DIR = Path(os.getenv("FL_UPLOAD_DIR", Path(tempfile.gettempdir()) / "firstlook_uploads"))
_uploads: dict[str, Path] = {}    # image ID -> scan uploaded by the clinician (replaces the stored one)
DB_VERSION = 3          # 3 = values from the ADNI export (M/F, genotype, GDS); no CDR, no research group

app = FastAPI(title="FirstLook inference service")
_lock = threading.Lock()          # one inference at a time on the GPU


def auth(authorization: str | None = Header(default=None)):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "missing or wrong token")


# ---------------------------------------------------------------- database
def _earliest(sub: pd.DataFrame, col: str):
    """First non-empty value by study date - the rule the cohort file was built with."""
    v = sub[col].dropna()
    return v.iloc[0] if len(v) else None


def _genotype(a1, a2):
    if a1 is None or a2 is None:
        return None
    a, b = sorted([int(float(a1)), int(float(a2))])
    return f"e{a}/e{b}"


def _db_tag(from_meta: bool) -> int:
    """Schema version plus whether the table was built from the ADNI export (last digit)."""
    return DB_VERSION * 10 + int(from_meta)


def build_db():
    """Patient table. Display values come from the ADNI export (idaSearch csv) with the same
    'earliest non-empty value per participant' rule that built df_cohort_final.pkl, so the doctor
    sees exactly what the model received (real units: M/F, APOE genotype). Missing model inputs were
    median-imputed for the model and are flagged. Not stored: Global CDR (disabled) and the ADNI
    research group, which IS the outcome label. The label is kept for the held-out fold split only
    and is never returned by the API."""
    df = pd.read_pickle(DATA_DIR / "df_cohort_final.pkl").reset_index(drop=True)
    folds = np.zeros(len(df), dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)     # as in train.py
    for f, (_, va) in enumerate(skf.split(np.zeros(len(df)), df["Label"].values)):
        folds[va] = f + 1
    raw = np.stack(df["Raw_Features"].values).astype(float)
    clin = np.stack([np.asarray(c[:7], dtype=float) for c in df["Clinical_Features"].values])
    clin[:, CDR] = 0.0                                                  # cohort mean in z-space

    meta = None
    if META_CSV.exists():
        ida = pd.read_csv(META_CSV, low_memory=False)
        ida["Study Date"] = pd.to_datetime(ida["Study Date"], errors="coerce")
        ida = ida.sort_values("Study Date", na_position="last", kind="stable")
        meta = {s: g for s, g in ida.groupby("Subject ID", sort=False)}
        print(f"Patient table from {META_CSV.name}")
    else:
        print(f"{META_CSV} not found - patient table from the cohort file only (coded values)")

    rows, mismatches = [], 0
    for i, r in df.iterrows():
        g = meta.get(r["Subject"]) if meta else None
        if g is not None:
            sex = _earliest(g, "Sex")
            gt = _genotype(_earliest(g, "APOE A1"), _earliest(g, "APOE A2"))
            vals = {k: _earliest(g, c) for k, c in [("age", "Age"), ("mmse", "MMSE Total Score"),
                    ("faq", "FAQ Total Score"), ("gds", "GDSCALE Total Score"),
                    ("npiq", "NPI-Q Total Score"), ("weight", "Weight")]}
            check = [vals["age"], vals["mmse"], vals["faq"], vals["gds"]]
            for a, b in zip(check, raw[i, [2, 3, 5, 6]]):
                if not ((a is None and np.isnan(b)) or (a is not None and abs(float(a) - b) < 1e-6)):
                    mismatches += 1
        else:
            sex = "M" if raw[i, 0] == 1 else "F"
            gt = None
            vals = {"age": raw[i, 2], "mmse": raw[i, 3], "faq": raw[i, 5], "gds": raw[i, 6],
                    "npiq": None, "weight": None}
        vals = {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
                for k, v in vals.items()}
        rows.append((r["Image Data ID"], r["Subject"], str(sex) if sex is not None else None, gt,
                     float(raw[i, 1]), vals["age"], vals["mmse"], vals["faq"], vals["gds"],
                     vals["npiq"], vals["weight"], int(r["Label"]), int(folds[i]),
                     ",".join(f"{v:.6f}" for v in clin[i])))
    if meta:
        print(f"Checked against the model inputs: {mismatches} mismatching values")

    con = sqlite3.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS patients")
    con.execute("""CREATE TABLE patients (image_id TEXT PRIMARY KEY, subject TEXT, sex TEXT,
                   apoe_genotype TEXT, apoe4 REAL, age REAL, mmse REAL, faq REAL, gds REAL,
                   npiq REAL, weight REAL, label INTEGER, heldout_fold INTEGER, clin_z TEXT)""")
    con.executemany(f"INSERT INTO patients VALUES ({','.join('?' * 14)})", rows)
    con.execute(f"PRAGMA user_version = {_db_tag(meta is not None)}")
    con.commit()
    con.close()


def db():
    if not DB_PATH.exists():
        build_db()
    else:
        c = sqlite3.connect(DB_PATH)
        tag = c.execute("PRAGMA user_version").fetchone()[0]
        c.close()
        # rebuild if the file is from an older schema, or the ADNI export has appeared since
        if tag < _db_tag(False) or (META_CSV.exists() and tag < _db_tag(True)):
            build_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def row(image_id: str):
    with db() as con:
        r = con.execute("SELECT * FROM patients WHERE image_id = ?", (image_id.strip().upper(),)).fetchone()
    if r is None:
        raise HTTPException(404, f"unknown image ID {image_id}")
    return r


def scan_path(r) -> Path:
    if r["image_id"] in _uploads:
        return _uploads[r["image_id"]]
    s = r["subject"]
    return DATA_DIR / f"sub-{s}" / f"sub-{s}_{r['image_id']}_desc-brain.nii.gz"


# ---------------------------------------------------------------- models and volumes
@lru_cache(maxsize=1)
def models():
    out = {}
    for f in range(1, 6):
        hits = sorted(MODELS_DIR.glob(f"Twostage_PRO_fold{f}_*.pth"))
        if not hits:
            raise RuntimeError(f"no checkpoint for fold {f} in {MODELS_DIR}")
        out[f] = load_fold(hits[0], DEVICE)
    return out


@lru_cache(maxsize=16)
def volume(image_id: str):
    """(raw volume or None, model-space tensor, mri_ok). A missing or unreadable scan becomes
    an all-zero volume, exactly as the training loader did; mri_ok tells the client."""
    r = row(image_id)
    try:
        raw = nib.load(str(scan_path(r))).get_fdata().astype(np.float32)
        proc, ok = preprocess_mri(raw), True
    except Exception:
        raw, proc, ok = None, np.zeros(TARGET_SHAPE, dtype=np.float32), False
    return raw, torch.from_numpy(proc).float()[None, None], ok


def on_gpu_or_cpu(fn):
    """Run fn(); if the GPU is out of memory (e.g. LM Studio holds most of it), free the cache,
    move the five models to the CPU for good and run again. CPU is slower but works."""
    global DEVICE
    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("GPU out of memory - switching the FirstLook models to the CPU", flush=True)
        DEVICE = "cpu"
        for m, _ in models().values():
            m.to("cpu")
        return fn()
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()     # give memory back to LM Studio between requests


def clin_tensor(r):
    return torch.tensor([float(v) for v in r["clin_z"].split(",")])[None]


# ---------------------------------------------------------------- endpoints
@app.get("/health", dependencies=[Depends(auth)])
def health():
    with db() as con:
        n = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    ms = models()
    gpu = None
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "free_gb": round(free / 2**30, 1),
               "total_gb": round(total / 2**30, 1),
               "peak_used_by_firstlook_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    return {"ok": True, "device": DEVICE, "gpu": gpu, "patients": n,
            "folds": {f: round(auc, 4) for f, (_, auc) in ms.items()}, "inputs": SERVED_INPUTS,
            "cdr": "disabled (cohort mean)"}


@app.get("/ids", dependencies=[Depends(auth)])
def ids(prefix: str = "", limit: int = 20):
    with db() as con:
        rows = con.execute("SELECT image_id FROM patients WHERE image_id LIKE ? ORDER BY image_id LIMIT ?",
                           (prefix.upper() + "%", limit)).fetchall()
    return [r[0] for r in rows]


@app.get("/patient/{image_id}", dependencies=[Depends(auth)])
def patient(image_id: str):
    r = row(image_id)
    out = {k: r[k] for k in ["sex", "apoe_genotype", "apoe4", "age", "mmse", "faq", "gds", "npiq", "weight"]}
    # Model inputs that were missing were median-imputed before z-scoring; the model saw that value,
    # the clinician sees that the measurement is missing.
    # Median-imputed for the model (cohort notebook: SimpleImputer(strategy="median"))
    out["missing"] = [k for k in ["age", "mmse", "faq", "gds"] if r[k] is None]
    # Not imputed but silently coded 0 by the cohort notebook: no APOE record -> "no e4",
    # no sex record -> female. The clinician must know the model assumed this.
    out["defaulted"] = ([k for k, c in [("apoe", "apoe_genotype"), ("sex", "sex")] if r[c] is None])
    return {"image_id": r["image_id"], **out, "mri_available": scan_path(r).exists(),
            "mri_source": "upload" if r["image_id"] in _uploads else "stored"}


@app.post("/upload/{image_id}", dependencies=[Depends(auth)])
async def upload(image_id: str, request: Request, name: str = "scan.nii.gz"):
    """Replace this patient's scan with one uploaded by the clinician (kept until the service restarts).
    The model expects a skull-stripped T1 like the training data."""
    r = row(image_id)
    iid = r["image_id"]
    data = await request.body()
    if not data or len(data) > 400 * 2**20:
        raise HTTPException(413, "empty file or larger than 400 MB")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / (iid + (".nii.gz" if name.lower().endswith(".gz") else ".nii"))
    path.write_bytes(data)
    try:
        shape = nib.load(str(path)).shape
        if len(shape) < 3 or min(shape[:3]) < 16:
            raise ValueError(f"shape {shape}")
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"not a readable 3D NIfTI volume ({type(exc).__name__})")
    with _lock:
        _uploads[iid] = path
        volume.cache_clear(); _predict.cache_clear(); _attribution.cache_clear()
    return {"image_id": iid, "shape": list(shape[:3]), "mri_source": "upload"}


@app.get("/mri/{image_id}.png", dependencies=[Depends(auth)])
def mri_png(image_id: str):
    raw, proc, ok = volume(row(image_id)["image_id"])
    png = render_png(raw if ok else proc[0, 0].numpy())
    return Response(png, media_type="image/png")


@lru_cache(maxsize=64)
def _predict(image_id: str):
    r = row(image_id)
    _, mri, ok = volume(image_id)
    clin = clin_tensor(r)
    # Stored scan: the fold whose validation set held this patient out never trained on it, so its
    # score is the main estimate. Uploaded scan: no fold has seen it, so all five are equal members
    # and the main estimate is their mean (Shapley values are averaged too, which is exact because
    # Shapley values are linear in the model output).
    uploaded = r["image_id"] in _uploads
    held = None if uploaded else int(r["heldout_fold"])
    t0 = time.perf_counter()

    def run():
        with torch.no_grad():
            m = mri.to(DEVICE); c = clin.to(DEVICE)
            folds = [{"fold": f, "prob": round(model(m, c).item(), 4), "held_out": f == held}
                     for f, (model, _) in models().items()]
            if held:
                return folds, *clinical_shapley(models()[held][0], m, c)
            parts = [clinical_shapley(model, m, c) for model, _ in models().values()]
            shap = {k: round(float(np.mean([p[0][k] for p in parts])), 4) for k in parts[0][0]}
            return folds, shap, round(float(np.mean([p[1] for p in parts])), 4)
    with _lock:
        folds, shap, base = on_gpu_or_cpu(run)
    shap.pop("Global CDR", None)            # exactly 0: CDR is fixed at the cohort mean
    probs = [f["prob"] for f in folds]
    main = folds[held - 1]["prob"] if held else round(float(np.mean(probs)), 4)
    return {"image_id": r["image_id"], "mri_ok": ok, "folds": folds, "held_out_fold": held,
            "estimate_mode": "ensemble" if uploaded else "held_out",
            "held_out_prob": main, "mean": round(float(np.mean(probs)), 4),
            "spread": round(max(probs) - min(probs), 4), "shap": shap, "shap_base": base,
            "inputs": SERVED_INPUTS, "cdr": "disabled (cohort mean)", "seconds": round(time.perf_counter() - t0, 2)}


@app.get("/predict/{image_id}", dependencies=[Depends(auth)])
def predict(image_id: str):
    return _predict(row(image_id)["image_id"])


@lru_cache(maxsize=16)
def _attribution(image_id: str) -> bytes:
    r = row(image_id)
    _, mri, ok = volume(image_id)
    if r["image_id"] in _uploads:           # uploaded scan: average the five models' maps
        members = [m for m, _ in models().values()]
    else:
        members = [models()[int(r["heldout_fold"])][0]]
    with _lock:
        cam = on_gpu_or_cpu(lambda: np.mean([smooth_gradcam(m, mri.to(DEVICE), clin_tensor(r).to(DEVICE),
                                                             samples=8 if len(members) == 1 else 3)
                                             for m in members], axis=0))
        cam = cam / (cam.max() + 1e-8)
    return render_png(mri[0, 0].numpy(), overlay=cam)


@app.get("/attribution/{image_id}.png", dependencies=[Depends(auth)])
def attribution(image_id: str):
    return Response(_attribution(row(image_id)["image_id"]), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    print(f"Data: {DATA_DIR.resolve()}  Models: {MODELS_DIR.resolve()}  Device: {DEVICE}")
    db().close()                      # build or check the patient table now, not on the first request
    models()                          # load the five checkpoints now (fails early if one is missing)
    uvicorn.run(app, host=os.getenv("FL_HOST", "0.0.0.0"), port=int(os.getenv("FL_PORT", "8008")))
