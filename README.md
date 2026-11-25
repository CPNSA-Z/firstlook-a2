# FirstLook: doctor-first review for Alzheimer's disease assessment (AD vs non-AD)

IIS Assignment 2 prototype · Yaowen Zhang (3120714) · 37809/37792 Intelligent Interactive Systems

Chinese version: [README_zh.md](README_zh.md)

**README prose word count: ~2,990** (headings and code blocks excluded; table text included).

> Research prototype, not for clinical use. The trained model separates Alzheimer's disease from non-AD participants (cognitively normal, subjective complaints, MCI); it is not an MCI-to-AD conversion model (item 14). The app runs against the **inference service** only (the real ADNI cohort and the five trained fold models, served by `inference_server/` on a local GPU PC; no ADNI data is in this repository). There is no offline or demo mode (item 17).

## Project status

| Part | Status |
|---|---|
| Streamlit interface: search page (image ID + optional T1 upload); Condition A (verdict + SHAP + MRI map baseline) and Condition B (doctor-first: patient table + MRI → own judgement, or name the main uncertainty and *View AI analysis* → AI analysis → final decision with an *Uncertain — refer* option) | Implemented |
| Structured explanation of the model result (local LLM, JSON schema; code checks every number, input name, direction and size against the model output and removes diagnosis/instruction language) | Implemented; unit-tested and run against the local LLM (item 17) |
| Inference service (`inference_server/`): patient table in local SQLite, MRI slices, five fold scores with the held-out fold as main estimate (uploaded scan: mean of the five), exact clinical Shapley values, smoothed Grad-CAM | Implemented; running on the GPU PC (RTX 5080) |
| Floating assistant (local LLM, any OpenAI-compatible server such as LM Studio): grounded to the record (and the model numbers after the judgement), told not to diagnose or simply agree; available in both conditions | Implemented; run against the local LLM |
| MRI panels: three orthogonal slices with attribution overlay | The patient's own scan (or the uploaded one) and real Grad-CAM |
| Rule-based checks in two separate sets: **input checks** (record only, shown in Step 1): failed MRI, age/MMSE outside training range, high depression score, inputs that were median-imputed or silently defaulted (APOE → non-carrier, sex → female); **uncertainty checks** (fold scores only, shown in Step 3): fold disagreement, borderline mean | Implemented, unit-tested; app-level test that the pre-judgement screen does not change when the fold scores change |
| If the local LLM fails | A template built from the numbers is shown, with the reason (unreachable, thinking used the whole token budget, or failed the checks); *Regenerate* asks again |

## Technology stack

- **Python 3.11+, Streamlit**: interface and session state (one file, `app.py`).
- **Local LLM** through any OpenAI-compatible server (I used LM Studio with a Qwen model; a private model works the same way): writes the structured explanation and answers the assistant's questions. Local, so no patient data leaves the network.
- **FastAPI + PyTorch** (`inference_server/`, on the GPU PC): patient table, MRI rendering, the five fold models, exact SHAP and Grad-CAM.
- **openai** Python client: talks to LM Studio; requests `json_schema` structured output.
- **Altair / pandas**: fold-score dot plot and attribution bars.
- Search-page background: MNI ICBM152 2009a template (© 1993–2009 Louis Collins, McConnell Brain Imaging Centre, MNI, McGill University; Fonov et al., 2011).
- **pytest** (+ Streamlit `AppTest`): tests for the rules, the validator and the no-leak property of Step 1. App-level tests use a stand-in for the inference service (`tests/fake_backend.py`, invented patients); it is not part of the app.
- Model (trained for this project; weights not in this repo): a 3D-CBAM CNN with FiLM fusion of 7 clinical features, trained by 5-fold cross-validation on 990 ADNI participants (scripts in `model_training/`). Saved fold checkpoints report best validation AUC 0.897, 0.872, 0.956, 0.773, 0.913 (mean ≈ 0.88).

## Running or inspecting the work

```bash
git clone https://github.com/CPNSA-Z/firstlook-a2.git && cd firstlook-a2
pip install -r requirements.txt
pytest -q                      # tests (several run the app headless against a stand-in service)
```

To run it: start `inference_server/` on the GPU PC (see its README), load a model in LM Studio, then on the app machine

```bash
export FIRSTLOOK_SERVER_URL="http://<gpu-pc-ip>:8008"  FIRSTLOOK_SERVER_TOKEN="<secret>"
export FIRSTLOOK_LLM_BASE_URL="http://<gpu-pc-ip>:1234/v1"  FIRSTLOOK_LLM_MODEL="<model id>"  FIRSTLOOK_LLM_API_KEY="<LM Studio token>"
streamlit run app.py
```

If the service does not answer, the app says so and stops; there is no fallback data. For screenshots, `FIRSTLOOK_PSEUDONYMS="CASE-001=<image id>,..."` shows and accepts pseudonyms instead of image IDs. Turn thinking off for the model in LM Studio (its API ignores `enable_thinking`); otherwise the app retries once with a larger token budget. Any other OpenAI-compatible endpoint works through the same variables. Supply your own key through `FIRSTLOOK_LLM_API_KEY`; no key is stored in the repository.

Suggested walkthrough: open a patient under Condition B, record a judgement that differs from the model, then one that agrees; open another and name an uncertainty to use *View AI analysis*; ask the assistant before and after; repeat one patient under Condition A. Each saved decision is appended to `logs/session_<id>.jsonl`.

Where things are: `model_training/` (training and SHAP scripts for the underlying model, with notes), `firstlook/scope.py` (input checks for Step 1, uncertainty checks for Step 3), `firstlook/llm.py` (prompt, schema, validator), `firstlook/model_adapter.py` (fold range), `firstlook/backend.py` (client for the inference service), `docs/` (system diagram), `scripts/figures/` (figure scripts).

## Development record

1. **Scope change: ED triage → memory clinic.** Assignment 1 proposed LLM triage in emergency departments. I moved to MCI conversion and trained a real model for it, so its SHAP outputs and failure history could drive the design instead of an invented model. The research question (uncertainty-first vs prescriptive + post-hoc SHAP) is unchanged.
2. **Reading my own pipeline as a source of failure cases.** Re-reading my `train.py` showed that an unreadable scan is replaced by an all-zero volume without warning. This became trap case C4 and the only *blocking* rule: the score is withheld even though the five folds agree (so spread alone would miss it). Test: `test_failed_mri_blocks_model_even_when_folds_agree`.
3. **Uncertainty from fold disagreement, not one probability.** The best fold reached AUC 0.956, but the five folds range from 0.77 to 0.96, so quoting the best one would overstate performance. Showing all five fold scores makes that variability visible. Bands use a 0.25 spread and a 0.35–0.65 borderline zone (my choice).
4. **Judgement before model.** Following Yildirim et al. (2026) and anchoring (Lecture 06), Step 2 locks the clinician's own judgement before Step 3 reveals anything; changing it afterwards requires a reason.
5. **LLM restricted to a non-core task.** Following Deng et al. (2025) and Lecture 08 (fluency is not evidence), the LLM only structures the record: the schema has no risk or diagnosis field, every cue needs a verbatim quote, and code checks the quote exists in the cited field, sets "did the model see this?" from `FIELD_REGISTRY` rather than trusting the LLM, and removes verdict language. Removed items stay visible. Two cached fixtures contain deliberate errors so the validator's behaviour can be inspected offline. What the real model got wrong, and the changes, are in item 17.
6. **Local model with labelled fallback.** LM Studio keeps patient text on the machine and needs no key. If the server does not answer, the app shows the cached output and says so, rather than failing silently (Lecture 04 feedback). (The cache was removed later; a labelled template replaces it, item 17.)
7. **Label fix.** The SHAP chart hid half its axis labels; fixed with a per-row height step (checked on screenshot).
8. **Clinical look and MRI panels.** After reviewing the first screenshots I decided the interface did not look like clinical software and that Condition A lacked the MRI heatmap that my A1 review criticises. I added a patient header bar, a step indicator and MRI panels. The image matters most in C4: Condition A draws an attribution map on an empty scan, while Condition B shows the blank volume beside the failed-load alert.
9. **Feature-name check (see item 14 for a correction).** Comparing my SHAP export with the cohort file actually used for training (`df_cohort_final.pkl`) showed that the SHAP script's feature names were shifted by one, so the SHAP labels were corrected.

10. **Step 1 leaked model output (found on review, fixed).** Reviewing Storyboard 1, I noticed that Step 1 showed "The five models disagree (range 0.47–0.81)" before the clinician's judgement was locked, and for borderline cases the mean score ("The average score (0.xx) is near the decision line"). This undid the anti-anchoring design: one function produced both record-based and score-based alerts, and the interface showed all of them at Step 1. The panel's step bar also showed Step 3, because it was cropped from a later screenshot.
    - *Change:* the rules are now two functions. `input_checks(record)` receives only the record and is shown at Step 1. `model_checks(view)` receives only the fold scores and is shown at Step 3, beside the fold dots. The failed-MRI rule stays at Step 1 (it describes the input, not a score) and still withholds the score at Step 3. This applies to Condition B only; Condition A keeps its confident verdict, so the C4 contrast is unchanged. The step bar now highlights the steps visible on screen (1–2, then 3–4). Figure 1 and all three storyboards were regenerated from new screenshots.
    - *Verification:* `test_input_checks_take_the_record_only` checks the function signature. `test_step1_screen_identical_whatever_the_fold_scores` runs the real app for every case with the true fold scores and with random ones, and asserts that the pre-judgement screen text is identical and contains none of the scores. I re-inserted the old behaviour to confirm the test catches it: all six cases failed.

11. **Reasoning model hung the live call.** With Qwen 3.6 in LM Studio the app showed nothing: the model spent ~3,000 reasoning tokens (~80 s), the client timed out and retried, and each Streamlit rerun started a new request. *Change:* thinking switched off (`enable_thinking: false` and `/no_think`), `max_tokens` capped, no automatic retries, and the call runs in a background job that survives reruns.
12. **Redesign after review: doctor-first with real data.** Reviewing the prototype, I decided the flow did not match how a clinician works. *Changes:* patients are looked up by **image ID** only (privacy); Step 1 shows the basic information from the database and the patient's own MRI slices; the model runs in the background but nothing from it is shown until the doctor chooses **Lower concern**, **Higher concern** or **View AI analysis**. The third button replaces "Uncertain": wording a doubt as "let me see the analysis" avoids making the doctor look unsure in front of the patient; it is logged as uncertain. If the doctor agrees with the model, a short confirmation is shown and the details are collapsed; if not (or the model is unsure), the full analysis opens with neutral wording, and changing the first judgement needs a reason. The analysis shows the held-out fold estimate, the five fold dots, exact SHAP bars, a Grad-CAM map and a local-LLM explanation written only from those numbers (any OpenAI-compatible model; I used Qwen). A floating assistant answers questions; before the judgement it is given the record only, so it cannot leak the result even if it ignores its instructions. *Verification:* `test_screen_before_judgement_identical_whatever_the_model_says`, `test_assistant_gets_no_model_result_before_judgement`, plus an end-to-end browser run against the inference service on 42 cohort scans.
13. **FiLM formula check.** Writing the inference service, I found two forms of the FiLM layer in my scripts: `img * gamma + beta` (`train.py`) and `img * (1 + gamma) + beta` (`run_shap.py`, `interpret_image.py`). On 40 of fold 3's own validation subjects, the first gives AUC 0.968 (training log 0.956), the second 0.902. The service uses the training form; the older SHAP figures came from a mismatched layer (noted in `model_training/NOTES.md`).
14. **Patient table rebuilt from the ADNI export; two earlier claims corrected.** Building the database from `idaSearch_10_06_2025.csv` (the export the cohort file was made from) with the cohort notebook's own rule (earliest non-empty value per participant) reproduces every model input exactly (990/990, including which values are missing). The doctor now sees real units (sex M/F, APOE genotype such as ε3/ε4, GDS, NPI-Q and weight) instead of codes. This check corrected two things: (a) the seventh model input is the **depression scale (GDS)**, not NPI-Q as I had first assumed, so `FIELD_REGISTRY`, the SHAP labels and the C3 description were fixed (sex coding confirmed: 1 = M); (b) the label is 1 for every participant in the ADNI **Alzheimer's disease group** and 0 for all others, including cognitively normal participants, so the model separates AD from non-AD rather than predicting MCI-to-AD conversion. The research group itself is not stored, because it is the label.
15. **Search page.** After trying the app, I moved patient lookup out of the side panel: the first screen now shows only an image-ID box, an optional T1 upload (.nii/.nii.gz, sent to the inference service and used instead of the stored scan) and a Find button, over a background made from the public MNI template. A pulsing-scan overlay shows while the record and MRI load. Study settings stay in the side panel; the page header is no longer hidden, because hiding it also removed the button that re-opens a collapsed side panel. *Tests:* search page shows only the form; unknown IDs stay on it; "New patient search" returns to it. The screens were then worded as a production tool would be: no dataset names, "random ID", "research prototype" or disclaimer badges on screen (the research status is stated here and in the report), and model internals such as fold numbers or rule sources are not shown.
16. **Review of the redesign against my own principles.** A review of the doctor-first version raised five issues; I accepted most and changed:
    - *Scope wording:* the screens, prompts and demo text now say Alzheimer's disease assessment (AD vs non-AD), matching what the model was trained to do.
    - *Uploaded scans:* no fold has seen an uploaded scan, so the main estimate becomes the mean of the five models (all shown as equal); SHAP is averaged across folds (exact, because Shapley values are linear in the output) and Grad-CAM averaged too. The caption notes that four folds still saw the patient's clinical values.
    - *LLM back on a structured, checked task:* the text explanation is now JSON (`summary`, `key_drivers`, `uncertainty_note`, `limitations`). Code removes any driver that names a non-input, has the wrong sign or the wrong size, and any sentence with a number that is not in the model output or with diagnosis/instruction language; removed items are listed under the explanation. If the LLM is unreachable, the same structure is built from the numbers by a template. The assistant shows its grounding limits at the top and is told not to agree with the doctor just to please.
    - *Cognitive forcing:* "View AI analysis" stays (the label does not make the doctor look unsure), but it unlocks only after the doctor names the main uncertainty (imaging, scores, history, missing data); the log records how often it was used, so the bypass rate can be measured. The final decision has a third option, *Uncertain — refer for further work-up*, so no verdict is forced.
    - *Assumed inputs:* reading the cohort notebook showed that missing APOE and sex were not imputed but coded 0 (non-carrier, female); Step 1 now warns about these and about median-imputed scores.
    - *Not changed:* I did not add a cache of real model outputs to the repository, because per-participant outputs derived from ADNI should not be distributed.
    *Tests:* validator, fallback, uncertainty gate, three-option decision, assumed-input alert.
    - *Offline review (later removed, item 17):* demo cases carried stored LLM explanations, re-validated each time they were shown.
17. **Real local run; offline mode removed.** Running the full flow on the GPU PC with the local LLM changed four things.
    - *The LLM "was not reachable" although it was running:* LM Studio ignores `enable_thinking: false` for this model, so all 2,000 tokens went to reasoning and the answer was empty. The app now tells this apart from an unreachable server, retries once with a larger budget, never uses a cut-off reasoning draft, and has a *Regenerate* button. The model name is no longer shown on screen.
    - *The checks were too strict and too loose:* they removed correct drivers written with the record's label ("FAQ functional score (0-30, …)") and the true sentence "NPI-Q did not influence the result", but let through "NPI-Q was filled in with cohort averages" (NPI-Q is not an input) and "the model cannot see the MRI scan" (it can; the LLM cannot). Non-input fields are now marked in the facts, input labels are matched, and both false claims are removed. The distance from 0.5 is now given to the LLM instead of being rejected when it worked it out.
    - *Storyboards:* all panels are now screenshots of this run in Chrome (pseudonymised IDs, real LLM text); in it every explanation sentence passed the checks, and the assistant once called two imputed inputs "recorded", which the checks do not cover (noted in the report). The screenshots show ADNI scans and per-participant values, so they and the storyboards are kept out of this repository (ADNI Data Use Agreement) and appear only in the report; earlier storyboards were removed from the repository for the same reason.
    - *No offline mode:* the six synthetic vignettes, the stored LLM outputs and the referral-note structurer (ADNI records have no notes) were removed, so everything on screen comes from the real models. Without the inference service the app now stops with a message. *Tests* run the app against a stand-in service in `tests/` (`test_no_offline_mode_without_the_service`, `test_failed_mri_withholds_the_result_in_the_app`, `test_claims_that_the_model_cannot_see_the_scan_are_removed`).

Commits: one commit per stage above. The repository was uploaded through the GitHub web interface, so the local commit history is reproduced in `docs/commit_history.txt`. Submitted version: tag `a2-submission`.

## AI assistance

- **Claude (Anthropic, via Claude Cowork)**: figure scripts, this README and a first draft of the report. I asked it to read my literature review, my training code and results, and the Weeks 1–4 slides first.
  - *What I checked or changed:* every vignette value against the training cohort table; that C4 really matches the zero-volume fallback in my training code; the per-fold AUCs against the checkpoint filenames; ran the tests and the app; corrected claims in the draft so that simulated parts are labelled.
- **LM Studio local model**: runtime component only (explanation and assistant). Its explanation is validated by code; removed items are shown.

## Additional files

- `docs/system_diagram.png|svg` (+ `_zh`) – Figure 1, from `scripts/figures/diagram_v2.py`. The storyboards (report Figures 2–4) are composed by `scripts/figures/make_sb_v3.py` from screenshots of a real local run (taken in Chrome with `FIRSTLOOK_PSEUDONYMS`); they contain ADNI scans, so they stay local (`.gitignore`) and are only in the report; `docs/commit_history.txt` – local development history (messages and dates).
- `model_training/` – four scripts that trained and explained the underlying model (data loader fallback, SHAP feature lists). Not run by FirstLook.
- No ZIP. Model checkpoints, ADNI data and screenshots showing ADNI scans are excluded (`.gitignore`): ADNI's Data Use Agreement forbids redistribution.
