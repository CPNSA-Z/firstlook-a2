import json

import pytest

from fake_backend import PATIENTS, FakeBackend
from firstlook.model_adapter import view_from_prediction
from firstlook.scope import input_checks, model_checks, model_usable

FAKE = FakeBackend()


def _record(iid):
    pt = FAKE.patient(iid)
    return {"age": pt.values.get("age"), "mmse": pt.values.get("mmse"), "gds": pt.values.get("gds"),
            "_missing": pt.missing, "_defaulted": pt.defaulted,
            "mri_status": "loaded" if pt.mri_available else "failed"}


def alert_fields(iid):
    return {a.field for a in input_checks(_record(iid)) + model_checks(view_from_prediction(FAKE.predict(iid)))}


def test_clear_cases_raise_no_alerts():
    assert alert_fields("P-HIGH") == set()
    assert alert_fields("P-LOW") == set()


def test_failed_mri_blocks_model_even_when_folds_agree():
    v = view_from_prediction(FAKE.predict("P-NOMRI"))
    assert not v.disagree                      # spread alone would not catch it
    assert not model_usable(input_checks(_record("P-NOMRI")))


def test_depression_and_disagreement_flagged():
    assert {"gds", "model:fold_probs"} <= alert_fields("P-SPLIT")


def test_out_of_range_age_flagged():
    assert "age" in alert_fields("P-OLD")


def test_borderline_case_reads_uncertain():
    assert view_from_prediction(FAKE.predict("P-MID")).band == "uncertain"


# ---------------- Step 1 must not leak any model output (anti-anchoring invariant)

def test_input_checks_take_the_record_only():
    import inspect
    assert list(inspect.signature(input_checks).parameters) == ["record"]


def test_disagreement_is_a_step3_alert_not_a_step1_alert():
    assert "gds" in {a.field for a in input_checks(_record("P-SPLIT"))}
    assert all(not a.field.startswith("model:") for a in input_checks(_record("P-SPLIT")))
    assert {a.field for a in model_checks(view_from_prediction(FAKE.predict("P-SPLIT")))} == {"model:fold_probs"}


APP = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py")


class _Inline:
    """Executor that runs jobs at once, so AppTest pages are deterministic."""
    def __init__(self, *a, **k):
        pass

    def submit(self, fn, *args):
        from concurrent.futures import Future
        f = Future()
        try:
            f.set_result(fn(*args))
        except Exception as exc:
            f.set_exception(exc)
        return f


def _app(monkeypatch, patients=None, chat_spy=None, backend_error=None):
    """AppTest against the fake inference service, jobs run inline, fresh caches."""
    import concurrent.futures
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    import firstlook.backend as fb

    def fake_get_backend():
        if backend_error:
            raise fb.BackendError(backend_error)
        return FakeBackend(patients)
    monkeypatch.setattr(fb, "get_backend", fake_get_backend)
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _Inline)
    if chat_spy is not None:
        import firstlook.llm as fl
        monkeypatch.setattr(fl, "chat", chat_spy)
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def _open(at, image_id, live_llm=False):
    if not live_llm:
        at.sidebar.radio[1].set_value("Off (template text)").run()
    at.text_input(key="search_id").set_value(image_id).run()
    [b for b in at.button if b.label == "Find patient"][0].click().run()
    assert not at.exception, at.exception
    assert any(b.label == "← New patient search" for b in at.button), "patient page did not open"
    return at


def _screen(at):
    parts = [e.value for kind in ("markdown", "warning", "error", "info", "caption", "success")
             for e in getattr(at, kind)]
    return "\n".join(str(x) for x in parts if not str(x).startswith("Session "))  # random id


def _view_ai(at, iid):
    at.radio(key=f"B_unsure_{iid}").set_value("Imaging unclear").run()
    _click(at, "View AI analysis")


def _click(at, label):
    [b for b in at.button if b.label == label][0].click().run()
    assert not at.exception, at.exception


def test_no_offline_mode_without_the_service(monkeypatch):
    at = _app(monkeypatch, backend_error="cannot reach http://gpu:8008")
    assert any("inference service is not reachable" in e.value for e in at.error)
    assert not any(b.label == "Find patient" for b in at.button)


@pytest.mark.parametrize("iid", list(PATIENTS))
def test_screen_before_judgement_identical_whatever_the_model_says(iid, monkeypatch):
    """Run the real app: everything shown before the judgement must be identical when every fold
    score is replaced, and must not contain any of the scores."""
    import copy

    def before(folds):
        patients = copy.deepcopy(PATIENTS)
        rec = list(patients[iid]); rec[3] = folds; patients[iid] = tuple(rec)
        return _screen(_open(_app(monkeypatch, patients), iid))

    real = PATIENTS[iid][3]
    a = before(real)
    b = before([0.01, 0.99, 0.02, 0.98, 0.5])
    assert "Step 3 · AI analysis" not in a and "AI estimate" not in a      # the page stops at Step 2
    assert a == b
    for p in set(real) | {sum(real) / 5, min(real), max(real)}:
        assert f"{p:.2f}" not in a


def test_assistant_gets_no_model_result_before_judgement(monkeypatch):
    calls = []

    def spy(history, patient_lines, pred=None):
        calls.append(pred)
        return "ok"

    at = _open(_app(monkeypatch, chat_spy=spy), "P-MID", live_llm=True)
    key = "P-MID|B"
    at.text_input(key=f"chat_q_{key}").input("Will this patient convert?")
    [b for b in at.button if b.label == "Send"][0].click().run()
    assert calls == [None]                        # before the judgement: record only
    _view_ai(at, "P-MID")
    at.text_input(key=f"chat_q_{key}").input("What pushed the estimate up?")
    [b for b in at.button if b.label == "Send"][0].click().run()
    assert calls[-1] is not None                  # after the reveal: model result included


def test_view_ai_analysis_is_neutral_and_needs_a_final_choice(monkeypatch):
    at = _open(_app(monkeypatch), "P-MID")
    assert [b for b in at.button if b.label == "View AI analysis"][0].disabled   # name the uncertainty first
    _view_ai(at, "P-MID")
    text = _screen(at)
    assert "AI estimate" in text and "reads this patient differently" not in text
    assert at.radio(key="B_final_P-MID").value is None
    assert [b for b in at.button if b.label == "Save decision"][0].disabled


def test_disagreement_shows_analysis_and_requires_reason_to_change(monkeypatch):
    at = _open(_app(monkeypatch), "P-HIGH")
    _click(at, "Lower concern")
    assert "reads this patient differently" in _screen(at)
    at.radio(key="B_final_P-HIGH").set_value("Higher concern").run()
    assert [b for b in at.button if b.label == "Save decision"][0].disabled      # reason required
    at.text_input(key="B_reason_P-HIGH").input("Imaging weighs more here").run()
    assert not [b for b in at.button if b.label == "Save decision"][0].disabled


def test_agreement_collapses_details(monkeypatch):
    at = _open(_app(monkeypatch), "P-HIGH")
    _click(at, "Higher concern")
    assert "points the same way" in _screen(at)
    assert any(e.label == "Show the AI analysis details" for e in at.expander)


def test_failed_mri_withholds_the_result_in_the_app(monkeypatch):
    at = _open(_app(monkeypatch), "P-NOMRI")
    assert "MRI did not load" in _screen(at)
    _click(at, "Lower concern")
    assert "AI estimate: 0.2" not in _screen(at)


def test_service_patients_have_no_names_and_one_held_out_fold():
    for iid in PATIENTS:
        assert "name" not in FAKE.patient(iid).values
        pred = FAKE.predict(iid)
        assert sum(f["held_out"] for f in pred.folds) == 1
        assert pred.held_out_prob == pred.folds[pred.held_out_fold - 1]["prob"]


def test_template_explanation_uses_only_given_numbers():
    from firstlook.llm import template_text
    pred = FAKE.predict("P-MID")
    assert f"{pred.held_out_prob:.2f}" in template_text(pred)


def test_cdr_is_never_shown_even_if_a_server_sends_it(monkeypatch):
    """Global CDR is disabled: an old server that still returns it must not reach the screen."""
    from firstlook.backend import FIELDS, MODEL_INPUTS, RemoteBackend

    class R:
        status_code = 200
        def json(self):
            return {"image_id": "I1", "age": 70, "sex": "F", "apoe_genotype": "e3/e4", "apoe4": 1.0,
                    "mmse": 27, "cdr": 0.5, "faq": 2, "missing": ["cdr", "gds"], "mri_available": True}

    b = RemoteBackend.__new__(RemoteBackend)
    monkeypatch.setattr(b, "_get", lambda *a, **k: R(), raising=False)
    pt = b.patient("I1")
    assert "cdr" not in FIELDS and "cdr" not in MODEL_INPUTS
    assert "cdr" not in pt.values and pt.missing == ["gds"]
    assert pt.values["sex"] == "F" and pt.values["apoe"] == "ε3/ε4"


def test_search_page_shows_only_the_search_form(monkeypatch):
    at = _app(monkeypatch)
    labels = {b.label for b in at.button}
    assert labels == {"Find patient"}
    assert len(at.text_input) == 1 + 1       # search box + reviewer ID in the side panel


def test_unknown_id_stays_on_search_page(monkeypatch):
    at = _app(monkeypatch)
    at.text_input(key="search_id").set_value("NOPE").run()
    [b for b in at.button if b.label == "Find patient"][0].click().run()
    assert any("No patient with image ID NOPE" in e.value for e in at.error)
    assert any(b.label == "Find patient" for b in at.button)


def test_new_patient_returns_to_search(monkeypatch):
    at = _open(_app(monkeypatch), "P-MID")
    [b for b in at.button if b.label == "← New patient search"][0].click().run()
    assert any(b.label == "Find patient" for b in at.button)


def test_final_decision_offers_uncertain_and_changing_needs_reason(monkeypatch):
    at = _open(_app(monkeypatch), "P-LOW")
    _click(at, "Lower concern")
    r = at.radio(key="B_final_P-LOW")
    assert "Uncertain — refer for further work-up" in r.options and r.value == "Lower concern"
    r.set_value("Uncertain — refer for further work-up").run()
    assert [b for b in at.button if b.label == "Save decision"][0].disabled


def test_missing_and_defaulted_inputs_are_flagged_in_step1():
    alerts = input_checks({"age": 75, "mmse": None, "gds": None, "mri_status": "loaded",
                           "_missing": ["mmse", "gds"], "_defaulted": ["apoe"]})
    msg = " ".join(a.message for a in alerts)
    assert "MMSE, GDS" in msg and "non-carrier" in msg


# ---------------- structured explanation: the LLM proposes, the system disposes
def _pred():
    from firstlook.backend import Prediction
    folds = [{"fold": i, "prob": p, "held_out": i == 2} for i, p in enumerate([0.48, 0.45, 0.44, 0.52, 0.49], 1)]
    return Prediction(True, folds, 2, 0.45, 0.476, 0.08,
                      {"APOE4": 0.0213, "FAQ": -0.0145, "Sex": -0.0053, "Age": -0.0011, "MMSE": -0.0022, "GDS": -0.0014},
                      0.4501, "service")


def test_explanation_validator_keeps_grounded_items_and_removes_the_rest():
    from firstlook.llm import result_facts, validate_explanation
    pred = _pred()
    facts = result_facts("Age (years): 76.9", pred)
    raw = {"summary": "The estimate is 0.45, near the middle.",
           "key_drivers": [
               {"input": "APOE4", "direction": "raises", "effect": 0.021, "text": "Carrier status raised it."},
               {"input": "FAQ", "direction": "raises", "effect": 0.015, "text": "FAQ raised it."},        # wrong sign
               {"input": "Hippocampus", "direction": "lowers", "effect": 0.02, "text": "Volume."},       # not an input
               {"input": "Sex", "direction": "lowers", "effect": 0.09, "text": "Sex lowered it."},       # wrong size
           ],
           "uncertainty_note": "The patient has Alzheimer's disease.",                                  # diagnosis
           "limitations": ["MMSE was filled in.", "Hippocampal volume is 2.7 ml.",                    # invented number
                           "Volume is 2.1 ml."]}             # 2.1 must not pass as 0.021 (APOE4 SHAP)
    e = validate_explanation(raw, pred, facts)
    assert [d["input"] for d in e.drivers] == ["APOE4"]
    assert e.summary and not e.uncertainty
    assert e.limitations == ["MMSE was filled in."]
    assert e.summary == "The estimate is 0.45, near the middle."
    reasons = " ".join(r["reason"] for r in e.removed)
    for why in ("wrong direction", "not a model input", "wrong size", "diagnosis", "not in the model output"):
        assert why in reasons


def test_explanation_may_not_treat_non_model_fields_as_inputs_and_distance_is_a_fact():
    from firstlook.llm import result_facts, validate_explanation
    pred = _pred()
    facts = result_facts("NPI-Q total: not recorded (not a model input)", pred)
    assert "from 0.5: 0.05" in facts
    raw = {"summary": "The estimate is 0.45, 0.05 from 0.5.", "key_drivers": [], "uncertainty_note": "The folds agree.",
           "limitations": ["NPI-Q was not recorded and likely filled in with cohort averages, which may affect accuracy.",
                           "Weight is not a model input.", "MMSE was filled in."]}
    e = validate_explanation(raw, pred, facts)
    assert e.summary and e.limitations == ["Weight is not a model input.", "MMSE was filled in."]
    assert "treats NPI-Q as a model input" in e.removed[0]["reason"]
    raw["limitations"] = ["The NPI-Q total was not recorded and thus did not influence the result.",
                          "Weight was not used as an input for this model's prediction."]
    assert len(validate_explanation(raw, pred, facts).limitations) == 2


def test_drivers_may_use_the_record_or_chart_label_of_an_input():
    from firstlook.llm import result_facts, validate_explanation
    pred = _pred()
    raw = {"summary": "The estimate is 0.45.", "uncertainty_note": "The folds agree.", "limitations": [],
           "key_drivers": [{"input": "FAQ functional score (0-30, higher = worse)", "direction": "lowers",
                            "effect": -0.0145, "text": "It lowered it."},
                           {"input": "MMSE (0-30)", "direction": "lowers", "effect": -0.002, "text": "It lowered it."},
                           {"input": "APOE4 carrier", "direction": "raises", "effect": 0.021, "text": "It raised it."},
                           {"input": "Weight (kg)", "direction": "lowers", "effect": -0.002, "text": "It lowered it."}]}
    e = validate_explanation(raw, pred, result_facts("Age (years): 76.9", pred))
    assert [d["input"] for d in e.drivers] == ["FAQ", "MMSE", "APOE4"]
    assert "not a model input (Weight (kg))" in e.removed[0]["reason"]


def test_explanation_falls_back_to_template_when_llm_down(monkeypatch):
    import firstlook.llm as llm
    monkeypatch.setattr(llm, "BASE_URL", "http://127.0.0.1:9/v1")
    e = llm.explain_result("Age (years): 76.9", _pred())
    assert e.source.startswith("template") and e.drivers[0]["input"] == "APOE4"


def _fake_openai(monkeypatch, replies):
    """Replace openai.OpenAI with a client that returns the given (content, reasoning, finish) replies in turn."""
    import openai
    from types import SimpleNamespace as NS
    seen = []

    class Client:
        def __init__(self, **kw):
            self.chat = NS(completions=NS(create=self.create))

        def create(self, **kw):
            seen.append(kw["max_tokens"])
            content, reasoning, finish = replies[min(len(seen), len(replies)) - 1]
            return NS(choices=[NS(finish_reason=finish, message=NS(content=content, reasoning_content=reasoning))],
                      usage=NS(completion_tokens=kw["max_tokens"]))
    monkeypatch.setattr(openai, "OpenAI", Client)
    return seen


def test_thinking_that_uses_the_whole_budget_is_retried_once_with_more_tokens(monkeypatch):
    import firstlook.llm as llm
    good = json.dumps({"summary": "The estimate is 0.45.", "uncertainty_note": "The folds agree.", "limitations": [],
                       "key_drivers": [{"input": "APOE4", "direction": "raises", "effect": 0.021, "text": "Raised it."}]})
    seen = _fake_openai(monkeypatch, [("", "Here's a thinking process: ...", "length"), (good, "", "stop")])
    e = llm.explain_result("Age (years): 76.9", _pred())
    assert e.source == "live" and seen == [llm.MAX_TOKENS, llm.RETRY_TOKENS]


def test_thinking_overrun_is_not_reported_as_unreachable(monkeypatch):
    import firstlook.llm as llm
    _fake_openai(monkeypatch, [("", '{"summary": "draft", "key_drivers": [', "length")])
    e = llm.explain_result("Age (years): 76.9", _pred())
    assert e.source == "template (ThinkingOverrun)"           # the cut-off reasoning draft is never used
    why = llm.failure_reason(e.source)
    assert "thinking" in why and "reached" not in why
    assert "could not be reached" in llm.failure_reason("template (APIConnectionError)")


def test_view_ai_choice_is_logged(monkeypatch, tmp_path):
    import firstlook.logger as lg
    monkeypatch.setattr(lg, "LOG_DIR", tmp_path)
    at = _open(_app(monkeypatch), "P-MID")
    _view_ai(at, "P-MID")
    at.radio(key="B_final_P-MID").set_value("Uncertain — refer for further work-up").run()
    [b for b in at.button if b.label == "Save decision"][0].click().run()
    import json as _json
    entry = _json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[-1])
    assert entry["first_judgement"].startswith("uncertain") and entry["uncertainty_source"] == "Imaging unclear"
    assert entry["final_judgement"].startswith("Uncertain")


def test_uploaded_scan_switches_the_main_estimate_to_the_five_model_mean(monkeypatch, tmp_path):
    """Server logic with stand-in models: stored scan -> held-out fold; uploaded scan -> mean of five."""
    pytest.importorskip("torch"); pytest.importorskip("fastapi"); pytest.importorskip("nibabel")
    import sys, torch
    sys.path.insert(0, str(__import__("pathlib").Path(APP).parent / "inference_server"))
    monkeypatch.setenv("FL_DATA_DIR", str(tmp_path)); monkeypatch.setenv("FL_MODELS_DIR", str(tmp_path))
    import server

    class Fake(torch.nn.Module):
        def __init__(self, p): super().__init__(); self.p = p
        def forward(self, m, c): return torch.tensor([self.p])
    probs = [0.2, 0.4, 0.6, 0.8, 0.5]
    monkeypatch.setattr(server, "models", lambda: {f: (Fake(p), 0.9) for f, p in enumerate(probs, 1)})
    monkeypatch.setattr(server, "row", lambda i: {"image_id": i, "heldout_fold": 2, "subject": "x",
                                                  "clin_z": ",".join(["0"] * 7)})
    monkeypatch.setattr(server, "volume", lambda i: (None, torch.zeros(1, 1, 2, 2, 2), True))
    monkeypatch.setattr(server, "clinical_shapley", lambda model, m, c: ({"Age": model.p / 10, "Global CDR": 0.0}, 0.5))
    server._predict.cache_clear(); server._uploads.clear()
    stored = server._predict("I1")
    assert stored["estimate_mode"] == "held_out" and stored["held_out_prob"] == 0.4
    assert [f["held_out"] for f in stored["folds"]] == [False, True, False, False, False]
    server._uploads["I1"] = tmp_path / "scan.nii.gz"; server._predict.cache_clear()
    up = server._predict("I1")
    assert up["estimate_mode"] == "ensemble" and up["held_out_fold"] is None
    assert abs(up["held_out_prob"] - 0.5) < 1e-6 and not any(f["held_out"] for f in up["folds"])
    assert abs(up["shap"]["Age"] - 0.05) < 1e-6 and "Global CDR" not in up["shap"]


def test_regenerate_button_asks_the_llm_again_and_model_name_is_not_shown(monkeypatch):
    import firstlook.llm as fl
    calls = []
    real = fl.explain_result
    monkeypatch.setattr(fl, "explain_result", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(fl, "BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(fl, "MODEL", "secret-model-name")
    at = _open(_app(monkeypatch), "P-MID", live_llm=True)
    _view_ai(at, "P-MID")
    assert len(calls) == 1
    [b for b in at.button if b.label == "↻ Regenerate"][0].click().run()
    assert len(calls) == 2
    assert "secret-model-name" not in _screen(at)


def test_claims_that_the_model_cannot_see_the_scan_are_removed():
    from firstlook.llm import result_facts, validate_explanation
    pred = _pred()
    raw = {"summary": "The estimate is 0.45.", "key_drivers": [], "uncertainty_note": "The folds agree.",
           "limitations": ["The model cannot see the MRI scan directly; it relies on processed features.",
                           "The explanation is written without seeing the scan.",
                           "The model's map of the MRI is coarse."]}
    e = validate_explanation(raw, pred, result_facts("Age (years): 76.9", pred))
    assert e.limitations == raw["limitations"][1:]
    assert "does not see the scan" in e.removed[0]["reason"]
