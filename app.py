"""FirstLook - Streamlit interface.

Run:  streamlit run app.py
Data source:
  FIRSTLOOK_SERVER_URL -> the FirstLook inference service on the GPU PC (real ADNI cohort from a local
                          database, real trained models, exact SHAP, Grad-CAM). There is no offline mode.
Patients are looked up by image ID only; no names are stored or shown.

Two conditions for the planned within-subject study:
  B  Doctor-first (proposed): patient table + MRI -> own judgement (lower / higher / "View AI analysis")
     -> AI analysis (estimate, fold spread, SHAP, attribution map, local-LLM explanation) -> final decision
  A  Prescriptive + SHAP (baseline): verdict first, explanation after
A floating assistant (local LLM) answers questions. Before the doctor's judgement it is given the
patient record only, never the model result, so it cannot leak the score.
"""
from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import altair as alt
import pandas as pd
import streamlit as st

from firstlook import ui_style as ui
from firstlook.config import DATA_DIR
from firstlook.backend import FIELDS, MODEL_INPUTS, SHAP_LABEL, BackendError, get_backend, patient_text
from firstlook.llm import MODEL, chat, explain_result, failure_reason, template_explanation
from firstlook.logger import log_decision
from firstlook.model_adapter import view_from_prediction
from firstlook.scope import input_checks, model_checks, model_usable

st.set_page_config(page_title="FirstLook", layout="wide", initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)

LOWER, HIGHER, VIEW_AI = "Lower concern", "Higher concern", "View AI analysis"
UNCERTAIN = "Uncertain — refer for further work-up"      # final decision only: no forced verdict
FINAL_OPTIONS = [LOWER, UNCERTAIN, HIGHER]
UNSURE_SOURCES = ["Imaging unclear", "Scores conflict", "Key information missing", "Other"]
GRADCAM_NOTE = ("Grad-CAM at the model's last convolution layer, smoothed over noisy copies of the scan. "
                "The map is coarse (low resolution): it shows roughly where the model "
                "looked, not a lesion. Red = pushed the estimate up.")

# Optional display pseudonyms for teaching and screenshots, e.g. FIRSTLOOK_PSEUDONYMS="CASE-001=I123456".
# The alias is shown and can be typed in the search box; the log keeps both.
PSEUDO = {a.strip().upper(): r.strip().upper() for a, r in
          (x.split("=", 1) for x in os.getenv("FIRSTLOOK_PSEUDONYMS", "").split(",") if "=" in x)}
ALIAS = {r: a for a, r in PSEUDO.items()}

ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex[:8])
ss.setdefault("progress", {})
ss.setdefault("active_id", "")      # patient currently open ("" = search page)
ss.setdefault("mri_png", {})        # image ID -> rendered slices (avoid re-rendering on every click)
ss.setdefault("upload_tag", {})     # image ID -> uploaded file name


@st.cache_resource
def _backend():
    return get_backend()      # raises BackendError if the inference service is not reachable (not cached)


@st.cache_resource
def _workers():
    """Worker pools and job table shared by all reruns, so slow calls are not restarted by clicks."""
    return {"model": ThreadPoolExecutor(max_workers=1), "llm": ThreadPoolExecutor(max_workers=1), "jobs": {}}


try:
    backend = _backend()
except BackendError as exc:
    st.error(f"**The FirstLook inference service is not reachable.**  \n{exc}  \n"
             "Start it on the GPU PC (inference_server/run_inference_server.ps1), then reload this page.")
    st.stop()


def job(key: tuple, pool: str, fn, *args) -> Future:
    jobs = _workers()["jobs"]
    if key not in jobs:
        jobs[key] = _workers()[pool].submit(fn, *args)
    return jobs[key]


@st.fragment(run_every=2)
def wait_for(fut: Future, message: str, t0: float):
    if fut.done():
        st.rerun()
    st.info(f"{message} {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- sidebar (study settings)
with st.sidebar:
    st.markdown("### Study settings")
    reviewer = st.text_input("Reviewer ID", value="R01")
    cond = st.radio("Interface condition", ["B: Doctor-first", "A: Prescriptive + SHAP"])
    cond_key = cond[0]
    llm_mode = st.radio("AI explanation and assistant", ["Live (local LLM)", "Off (template text)"],
                        help="Live uses the local LLM on the OpenAI-compatible server set in FIRSTLOOK_LLM_BASE_URL.")
    LLM_ON = llm_mode.startswith("Live")
    st.divider()
    st.caption(f"Data: {backend.name}.")
    st.caption(f"Session {ss.session_id}. Decisions are written to logs/.")


def mri_slices(iid: str) -> bytes:
    if iid not in ss.mri_png:
        ss.mri_png[iid] = backend.mri_png(iid)
    return ss.mri_png[iid]


# ---------------------------------------------------------------- search page
def search_page():
    import base64
    bg = base64.b64encode((DATA_DIR / "ui" / "landing_bg.jpg").read_bytes()).decode()
    st.markdown(ui.landing_css(bg), unsafe_allow_html=True)
    st.markdown(ui.hero(), unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid, st.container(key="fl_search"):
        st.text_input("Patient image ID", key="search_id",
                      placeholder="Image ID",
                      help="Patients are identified by their MRI image ID, not by name (privacy).")
        up = st.file_uploader("T1 MRI scan (optional, .nii / .nii.gz)", type=["nii", "gz"], key="search_mri",
                              help="Leave empty to use the scan stored for this ID. The model expects a "
                                   "skull-stripped T1 like its training data.")
        find = st.button("Find patient", type="primary", width="stretch")
        msg = st.empty()
    if not find:
        return
    iid = ss.get("search_id", "").strip().upper()
    iid = PSEUDO.get(iid, iid)
    if not iid:
        msg.warning("Enter an image ID.")
        return
    overlay = st.empty()
    overlay.markdown(ui.loader("Loading patient record and MRI..."), unsafe_allow_html=True)
    try:
        backend.patient(iid)
        if up is not None:
            overlay.markdown(ui.loader(f"Uploading {up.name}..."), unsafe_allow_html=True)
            backend.upload_mri(iid, up.name, up.getvalue())
            ss.upload_tag[iid] = up.name
            ss.mri_png.pop(iid, None)
        overlay.markdown(ui.loader("Rendering MRI slices..."), unsafe_allow_html=True)
        try:
            mri_slices(iid)
        except (KeyError, BackendError, FileNotFoundError):
            pass                                   # shown as "no image" on the next page
    except KeyError:
        overlay.empty()
        msg.error(f"No patient with image ID {iid}.")
        return
    except BackendError as exc:
        overlay.empty()
        msg.error(str(exc))
        return
    ss.active_id = iid
    st.rerun()


if not ss.active_id:
    search_page()
    st.stop()

image_id = ss.active_id
try:
    patient = backend.patient(image_id)
except (KeyError, BackendError) as exc:
    ss.active_id = ""
    st.error(f"Could not load {image_id}: {exc}")
    st.stop()


def new_patient():
    ss.active_id = ""
    ss.search_id = ""


pkey = f"{image_id}|{cond_key}"
if pkey not in ss.progress:
    ss.progress[pkey] = {"t_open": time.time(), "step": 1, "chat": []}
p = ss.progress[pkey]


def fmt(v):
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def patient_lines() -> str:
    return patient_text(patient)


record_for_checks = {"age": patient.values.get("age"), "mmse": patient.values.get("mmse"),
                     "gds": patient.values.get("gds"), "_missing": patient.missing,
                     "_defaulted": patient.defaulted,
                     "mri_status": "loaded" if patient.mri_available else "failed"}
in_alerts = input_checks(record_for_checks)          # record only: safe before the judgement

# Start the model analysis in the background straight away (hidden until the reveal).
analysis = job(("model", backend.name, image_id, ss.upload_tag.get(image_id)), "model",
               lambda i: {"pred": backend.predict(i), "map": backend.attribution_png(i)}, image_id)

st.markdown(ui.patient_header(ALIAS.get(image_id, image_id), fmt(patient.values.get("age")), fmt(patient.values.get("sex")) or "?",
                              "Condition " + cond_key), unsafe_allow_html=True)
st.button("← New patient search", on_click=new_patient)


# ---------------------------------------------------------------- shared views
def show_alerts(alerts):
    for al in alerts:
        (st.error if al.level == "block" else st.warning)(f"**{al.message}**  \n{al.action}")


def show_patient():
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("**Basic information** (from the clinic database)")
        rows = []
        for fid, label in FIELDS.items():
            if fid not in patient.values and fid not in patient.missing:
                continue
            v = fmt(patient.values.get(fid))
            rows.append({"Field": label,
                         "Value": v if v is not None else ("not recorded (filled in for the model)"
                                                           if fid in MODEL_INPUTS else "not recorded"),
                         "Model input": "yes" if fid in MODEL_INPUTS else "no"})
        st.table(pd.DataFrame(rows).set_index("Field"))
        st.caption("Earliest recorded value for this patient (the value the model used); it may come from "
                   "an earlier visit than the scan. Global CDR is not used: it is part of the diagnostic criteria, so the model receives "
                   "a neutral value in its place.")
    with right:
        chip = (ui.mri_chip(False, "NOT AVAILABLE") if not patient.mri_available else
                ui.mri_chip(True, f"Uploaded · {ss.upload_tag.get(image_id, 'scan')}")
                if patient.mri_source == "upload" else ui.mri_chip(True, "Stored scan · skull-stripped"))
        st.markdown("**T1 MRI** &nbsp; " + chip, unsafe_allow_html=True)
        try:
            st.image(mri_slices(image_id), width="stretch")
            st.markdown('<div class="fl-note">Axial, coronal and sagittal slices through the centre of this '
                        "patient's scan.</div>", unsafe_allow_html=True)
        except (KeyError, BackendError, FileNotFoundError):
            st.warning("No image could be loaded for this ID.")


def fold_chart(pred):
    ens = pred.estimate_mode == "ensemble"
    df = pd.DataFrame([{"Fold": f"Model {f['fold']}", "Score": f["prob"],
                        "Model": "main estimate" if f["held_out"] else "other models"}
                       for f in pred.folds])
    if ens:   # uploaded scan: no model has seen it, all five are equal members; their mean is the estimate
        df["Model"] = "five models (equal)"
        df = pd.concat([df, pd.DataFrame([{"Fold": "Mean", "Score": pred.held_out_prob, "Model": "main estimate"}])])
    dots = alt.Chart(df).mark_circle(size=170).encode(
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Estimated probability of Alzheimer's disease"),
        color=alt.Color("Model:N", scale=alt.Scale(
                            domain=["main estimate", "five models (equal)"] if ens else ["main estimate", "other models"],
                            range=["#c0392b", "#5b7fa6"] if ens else ["#c0392b", "#9fb3c8"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Fold", "Score", "Model"]).properties(height=80)
    rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    return dots + rule


def shap_chart(shap: dict):
    df = pd.DataFrame([{"Input": SHAP_LABEL.get(k, k), "Contribution": v} for k, v in shap.items()]) \
        .sort_values("Contribution")
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("Contribution:Q", title="Push on the estimate (−  lower | higher  +)"),
        y=alt.Y("Input:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelFontSize=12)),
        color=alt.condition("datum.Contribution > 0", alt.value("#c0504d"), alt.value("#4f81bd")),
        tooltip=["Input", alt.Tooltip("Contribution:Q", format="+.3f")],
    ).properties(height=alt.Step(28))


def render_explanation(exp):
    if exp.summary:
        st.write(exp.summary)
    for d in exp.drivers:
        arrow = "▲" if d["value"] > 0 else "▼"
        st.markdown(f"- {arrow} **{SHAP_LABEL.get(d['input'], d['input'])}** ({d['value']:+.3f}): {d['text']}")
    if exp.uncertainty:
        st.markdown(f"*{exp.uncertainty}*")
    for t in exp.limitations:
        st.caption("Limitation: " + t)
    if exp.removed:
        with st.expander(f"{len(exp.removed)} AI statement(s) removed by checks"):
            for r in exp.removed:
                item = r["item"].get("text", r["item"]) if isinstance(r["item"], dict) else r["item"]
                st.markdown(f"- “{item}” — {r['reason']}")


def show_explanation(pred):
    rnd = p.setdefault("explain_round", 0)                                # bumped by "Regenerate"
    fut = None
    if LLM_ON:
        fut = job(("explain", image_id, backend.name, MODEL, ss.upload_tag.get(image_id), rnd), "llm",
                  explain_result, patient_lines(), pred)
    h1, h2 = st.columns([5, 1], vertical_alignment="center")
    h1.markdown("**In words** (written by the local LLM from the numbers above and checked against them; "
                "it does not see the scan)")
    if LLM_ON and h2.button("↻ Regenerate", key=f"regen_{pkey}", disabled=not fut.done(), width="stretch",
                            help="Ask the local LLM to write the explanation again (e.g. after a failure)."):
        p["explain_round"] = rnd + 1
        st.rerun()
    if not LLM_ON:
        render_explanation(template_explanation(pred))
        st.caption("Built from the numbers by a template (local LLM switched off).")
        p["explanation_removed"] = 0
        return None
    if not fut.done():
        wait_for(fut, "Writing the explanation...", time.time())
        return None
    exp = fut.result()
    render_explanation(exp)
    st.caption("Every sentence was checked: numbers must match the model output, inputs must exist, directions "
               "must match the chart, no diagnosis or instruction." if exp.source == "live" else
               f"A template built from the numbers is shown because {failure_reason(exp.source)}. "
               "Press Regenerate to try again.")
    p["explanation_removed"] = len(exp.removed)
    return exp


def ai_panel(pred, view, explain=True):
    ens = pred.estimate_mode == "ensemble"
    st.markdown(ui.ai_card(pred.held_out_prob, pred.held_out_fold, view.band, ens), unsafe_allow_html=True)
    show_alerts(model_checks(view))
    c1, c2 = st.columns([1, 1.15])
    with c1:
        st.markdown("**Agreement between models**")
        st.altair_chart(fold_chart(pred), width="stretch")
        st.caption(("Five independently trained models. None has seen the uploaded scan, so all five count "
                    "equally and their mean (red) is the estimate. The clinical values come from the record, "
                    "which four of them saw in training." if ens else
                    "Five independently trained models. The red one never saw this patient in training and gives "
                    "the main estimate; the grey ones show how far the models agree."))
        st.markdown("**What moved the estimate: clinical inputs (exact SHAP" + (", mean of five models" if ens else "") + ")**")
        st.altair_chart(shap_chart(pred.shap), width="stretch")
        if pred.shap_base is not None:
            st.caption(f"Starting point {pred.shap_base:.2f} = this MRI with cohort-average clinical values. "
                       "Bars add up to the estimate. This shows what moved the model, not why the patient is ill.")
    with c2:
        st.markdown("**Where the model looked in the MRI**")
        st.image(job_result["map"], width="stretch")
        st.markdown(f'<div class="fl-note">{GRADCAM_NOTE}</div>', unsafe_allow_html=True)
    if explain:
        show_explanation(pred)


def save(final: str, reason: str, extra: dict):
    pred = job_result["pred"] if job_result else None
    entry = {"session": ss.session_id, "reviewer": reviewer, "condition": cond_key, "image_id": image_id,
             "display_id": ALIAS.get(image_id, image_id),
             "data_source": backend.name, "final_judgement": final, "reason": reason,
             "seconds_to_final": round(time.time() - p["t_open"], 1),
             "ai_estimate": round(pred.held_out_prob, 4) if pred else None,
             "ai_fold_spread": round(pred.spread, 4) if pred else None,
             "input_alerts": [a.field for a in in_alerts], "assistant_questions": sum(
                 1 for m in p["chat"] if m["role"] == "user"), **extra}
    path = log_decision(entry, ss.session_id)
    p["saved"] = entry
    st.success(f"Saved to {path.name}. You can revise and save again; every save is kept.")


# ---------------------------------------------------------------- floating assistant
revealed = cond_key == "A" or p["step"] >= 3
with st.container(key="fl_chat"):
    with st.popover("Ask the AI assistant"):
        st.info("Answers only from this patient's record" + (" and the model's numbers" if revealed else "")
                + ". It cannot see the scan and does not diagnose. Check anything important against the record.")
        st.caption("Local LLM. " + ("It has the patient record and the AI analysis." if revealed else
                   "It has the patient record only; the AI analysis is hidden until you record your judgement."))
        if not LLM_ON:
            st.info("Assistant is switched off in the sidebar.")
        for m in p["chat"]:
            st.chat_message(m["role"]).write(m["content"])
        if p["chat"] and p["chat"][-1]["role"] == "user":      # question waiting for an answer
            pred_ctx = None
            if revealed and analysis.done() and not analysis.exception():
                pred_ctx = analysis.result()["pred"]
            with st.spinner("Thinking..."):
                try:
                    answer = chat(p["chat"], patient_lines(), pred_ctx)
                except Exception as exc:
                    answer = f"(The assistant could not be reached: {type(exc).__name__}.)"
            p["chat"].append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)

        def _ask(key=f"chat_q_{pkey}"):
            q = ss.get(key, "").strip()
            if q:
                p["chat"].append({"role": "user", "content": q})

        with st.form(f"chat_{pkey}", clear_on_submit=True):
            st.text_input("Your question", key=f"chat_q_{pkey}", disabled=not LLM_ON,
                          placeholder="e.g. Which values are missing for this patient?")
            st.form_submit_button("Send", disabled=not LLM_ON, on_click=_ask)

def get_analysis():
    """Returns the analysis dict, or None after drawing a wait/error message."""
    if not analysis.done():
        wait_for(analysis, "Running the model on the GPU PC...", p.setdefault("t_wait", time.time()))
        return None
    if analysis.exception():
        st.error(f"The model could not run: {analysis.exception()}")
        if st.button("Retry"):
            _workers()["jobs"].pop(("model", backend.name, image_id, ss.upload_tag.get(image_id)), None)
            st.rerun()
        return None
    return analysis.result()


# ======================================================== CONDITION A (baseline)
if cond_key == "A":
    job_result = get_analysis()
    if job_result is None:
        st.stop()
    pred = job_result["pred"]
    view = view_from_prediction(pred)
    ai_panel(pred, view, explain=False)     # baseline: verdict + post-hoc SHAP, no text explanation
    with st.expander("Patient information and MRI"):
        show_patient()
    st.markdown(ui.section("Your decision"), unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    final = d1.radio("Your assessment", FINAL_OPTIONS, index=None, key=f"A_final_{image_id}")
    agree = d2.radio("Relative to the AI", ["Accept AI estimate", "Override"], index=None, key=f"A_agree_{image_id}")
    reason = st.text_input("Reason (optional)", key=f"A_reason_{image_id}")
    if st.button("Save decision", type="primary", disabled=final is None or agree is None):
        save(final, reason, {"accepted_ai": agree.startswith("Accept"), "estimate_mode": pred.estimate_mode})
    st.stop()

# ======================================================== CONDITION B (doctor-first)
job_result = None
st.markdown(ui.stepper({1, 2} if p["step"] <= 2 else {3, 4}), unsafe_allow_html=True)

st.markdown(ui.section("Step 1 · Patient information and MRI"), unsafe_allow_html=True)
show_alerts(in_alerts)
show_patient()

st.markdown(ui.section("Step 2 · Your judgement"), unsafe_allow_html=True)
if p["step"] <= 2:
    st.caption("Make your own read first. If you are not sure yet, say where the main uncertainty lies, "
               "then view the AI analysis.")
    conf = st.slider("Confidence (for lower / higher)", 1, 5, 3, key=f"B_conf_{image_id}")
    unsure = st.radio("Not sure yet? Main uncertainty:", UNSURE_SOURCES, index=None, horizontal=True,
                      key=f"B_unsure_{image_id}")
    b1, b2, b3 = st.columns(3)
    choice = None
    if b1.button(LOWER, width="stretch"):
        choice = LOWER
    if b2.button(HIGHER, width="stretch"):
        choice = HIGHER
    if b3.button(VIEW_AI, width="stretch", type="primary", disabled=unsure is None,
                 help="Choose the main uncertainty first: naming it is part of your own read."):
        choice = VIEW_AI
    if choice:
        p.update(step=3, own=choice, own_conf=None if choice == VIEW_AI else conf, t_own=time.time(),
                 unsure_source=unsure if choice == VIEW_AI else None)
        st.rerun()
    st.stop()

first = p["own"]
st.caption(f"Your first read: **{'Not sure yet (' + p['unsure_source'] + ')' if first == VIEW_AI else first}**"
           + (f" (confidence {p['own_conf']}/5)" if p["own_conf"] else "")
           + ". Recorded before the AI analysis was shown.")

# ---------- Step 3: AI analysis
st.markdown(ui.section("Step 3 · AI analysis"), unsafe_allow_html=True)
job_result = get_analysis()
if job_result is None:
    st.stop()
pred = job_result["pred"]
view = view_from_prediction(pred)
usable = model_usable(in_alerts) and pred.mri_ok      # a missing scan withholds the result (Condition B)
ai_dir = HIGHER if pred.held_out_prob >= 0.5 else LOWER
agree = None if first == VIEW_AI or view.band == "uncertain" else first == ai_dir

if not usable:
    st.error("AI result withheld: the MRI did not load, so the model saw a blank volume. "
             "Judge from the record; re-run when the scan is available.")
    if st.checkbox("Show the invalid result anyway (audit only)", key=f"B_force_{image_id}"):
        ai_panel(pred, view)
elif first == VIEW_AI:
    ai_panel(pred, view)
elif agree is None:
    st.markdown('<div class="fl-differ"><b>The model is unsure about this patient</b> (estimate near the '
                'decision line, or the fold models disagree). Your own reading carries more weight here; '
                'the details are below.</div>', unsafe_allow_html=True)
    ai_panel(pred, view)
elif agree:
    st.markdown(f'<div class="fl-agree"><b>The AI analysis points the same way as your judgement</b> '
                f'(estimate {pred.held_out_prob:.2f}).</div>', unsafe_allow_html=True)
    show_alerts(model_checks(view))
    with st.expander("Show the AI analysis details"):
        ai_panel(pred, view)
else:
    st.markdown('<div class="fl-differ"><b>The AI analysis reads this patient differently.</b> '
                'The model and the record can weigh different things, so this is worth a second look. '
                'What drove the model is shown below; the decision stays yours.</div>', unsafe_allow_html=True)
    ai_panel(pred, view)

# ---------- Step 4: final decision
st.markdown(ui.section("Step 4 · Final decision"), unsafe_allow_html=True)
opts = FINAL_OPTIONS
final = st.radio("Your assessment", opts, index=opts.index(first) if first in opts else None,
                 key=f"B_final_{image_id}", horizontal=True)
changed = first in (LOWER, HIGHER) and final is not None and final != first
reason = st.text_input("Reason" + (" (required: you changed your first judgement)" if changed else " (optional)"),
                       key=f"B_reason_{image_id}")
if st.button("Save decision", type="primary", disabled=final is None or (changed and not reason.strip())):
    save(final, reason, {
        "first_judgement": "uncertain (viewed AI analysis)" if first == VIEW_AI else first,
        "uncertainty_source": p.get("unsure_source"), "estimate_mode": pred.estimate_mode,
        "explanation_items_removed": p.get("explanation_removed"),
        "explanation_regenerated": p.get("explain_round", 0),
        "first_confidence": p["own_conf"], "seconds_to_first": round(p["t_own"] - p["t_open"], 1),
        "ai_direction": ai_dir, "agreed_with_ai_at_first": agree, "changed_after_ai": changed,
        "ai_withheld": not usable,
    })
