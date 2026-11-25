# FirstLook inference service

Runs on the GPU PC (the same machine as LM Studio). It serves the real cohort to the FirstLook app,
which only ever sends an **image ID**. No ADNI data is stored in this repository.

| What it needs | Where | Setting |
|---|---|---|
| `df_cohort_final.pkl` and the `sub-*/…_desc-brain.nii.gz` volumes | the preprocessed `Clean_Data` folder | `FL_DATA_DIR` |
| the five checkpoints `Twostage_PRO_fold{1..5}_*.pth` | any folder | `FL_MODELS_DIR` |
| the ADNI export `idaSearch_10_06_2025.csv` (for the patient table in real units) | any path | `FL_META` (default: inside `FL_DATA_DIR`) |
| a shared secret the app must send | you choose it | `FL_TOKEN` |

On first start it builds `firstlook_patients.db` (SQLite) next to the data (`FL_DB` to move it). Values
come from the ADNI export with the cohort notebook's rule (earliest non-empty value per participant),
and are checked against the model inputs at build time (the log prints the number of mismatches; 0
expected). Not stored: the ADNI research group. The outcome label is stored for the
held-out fold split only and is never served. Without the csv the table falls back to coded values.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8 build (needed for RTX 50-series)
pip install -r requirements.txt
$env:FL_DATA_DIR="D:\FirstLook\Clean_Data"; $env:FL_MODELS_DIR="D:\FirstLook\models"; $env:FL_TOKEN="<secret>"
python server.py        # listens on 0.0.0.0:8008
```

Or use `run_inference_server.ps1` (edit the three paths/secret at the top). Allow the port through
Windows Firewall once (administrator PowerShell):
`netsh advfirewall firewall add rule name="FirstLook 8008" dir=in action=allow protocol=TCP localport=8008`

On the laptop running the app: `FIRSTLOOK_SERVER_URL=http://<gpu-pc-ip>:8008`, `FIRSTLOOK_SERVER_TOKEN=<secret>`.

## GPU memory

The five fold models are about 1 MB each and one forward pass needs well under 0.1 GB of activations,
so the service needs roughly 1 GB of GPU memory in total (mostly the CUDA context). It can share the
card with LM Studio. If the GPU runs out of memory anyway, the service moves its models to the CPU and
carries on (slower: a few seconds per patient, Grad-CAM about 10 s). `/health` reports free GPU memory.
Set `$Device = "cpu"` in the run script to keep the whole card for LM Studio.

## What it computes

* **Five fold scores.** Fold *k* is assigned with `StratifiedKFold(5, shuffle=True, random_state=42)` on
  the pkl order, as in training. The fold whose validation set contained the patient never trained on
  them: its score is the **main estimate**; the other four are shown for spread only (they saw the
  patient in training, so they are optimistic).
* **Uploaded scan** (`POST /upload/{image_id}`): no fold has seen it, so the main estimate is the mean of
  the five, SHAP values are averaged over the five (exact for the mean model) and Grad-CAM maps averaged.
* **Exact Shapley values** for the six clinical inputs in use (computed exactly over all coalitions; an absent input is set to
  the cohort mean, the image embedding is held fixed). They add up to the estimate minus the
  "cohort-average clinical values" baseline.
* **Smoothed Grad-CAM** of the held-out fold model at `conv5` (8 noisy copies). Coarse resolution.

## Checks done when this was written

* Architecture and preprocessing are copied from `model_training/train.py`; checkpoints load with
  `strict=True`.
* FiLM fusion is `img * gamma + beta` (as in `train.py`). On 40 subjects of fold 3's own validation set
  this gives AUC 0.968 (training log: 0.956); the `img * (1 + gamma) + beta` variant used in the old
  `run_shap.py` / `interpret_image.py` gives 0.902, so those scripts used a mismatched layer.

## Known limitations

* **What the model predicts:** Label = 1 for the ADNI Alzheimer's disease group, 0 for everyone else
  (CN, SMC, EMCI, LMCI, MCI). It separates AD from non-AD; it is not an MCI-to-AD conversion model.
* Clinical inputs in use: Sex (1 = M), APOE4 carrier, Age, MMSE, FAQ, **GDS**.
  Missing values were median-imputed before training; the app shows them as
  "not recorded (filled in for the model)".
