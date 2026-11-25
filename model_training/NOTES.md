# Model training scripts

These scripts trained and explained the 3D-CBAM + FiLM model behind FirstLook. FirstLook does not import or run them; they need ADNI data and a GPU cluster, neither of which is in this repository. They are kept as evidence for design decisions.

| File | Why it is here |
|---|---|
| `train.py` | Final two-stage 3D-CBAM + FiLM training script. Line 129: an unreadable scan is replaced with `np.zeros(...)` without warning. This is the origin of trap case C4 and the blocking rule in `firstlook/scope.py`. |
| `run_shap.py` | Final clinical SHAP script. Its feature names (…FAQ, NPI-Q, GDSCALE) are shifted. Rebuilding the features from the ADNI export (exact match, 990/990) shows that the depression input is **GDS** (an earlier note here said NPI-Q; that was wrong). |
| `run_shap_990_early_8_inputs.py` | Earlier SHAP script for an 8-input model, kept for the record. |
| `interpret_image.py` | Smooth-SHAP MRI attribution (FirstLook uses Grad-CAM instead). |

Saved fold checkpoints (not included: trained on ADNI data): `Twostage_PRO_fold1..5`, best validation AUC 0.8968, 0.8717, 0.9561, 0.7729, 0.9127 (mean ≈ 0.882). Their first clinical layer has weight shape 32 × 7.

**FiLM mismatch.** `train.py` fuses with `img * gamma + beta`; `run_shap.py` and `interpret_image.py` rebuild the layer as `img * (1 + gamma) + beta`. Checked on 40 subjects of fold 3's validation set: AUC 0.968 with the training form, 0.902 with the other. Figures produced by those two scripts therefore came from a slightly different model. `inference_server/fl_model.py` uses the training form.

**Label.** The cohort notebook sets Label = 1 when the participant's ADNI research group contains "AD" and 0 otherwise (CN, SMC, EMCI, LMCI, MCI). The checkpoints therefore separate the AD group from everyone else; they are not MCI-to-AD conversion models.
