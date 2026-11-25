"""Local LLM: structured explanation of the model result, and the floating assistant.

The LLM never judges the patient. It (1) turns the model's numbers into a short explanation and
(2) answers the doctor's questions. The model proposes; this module disposes (Lecture 12):
  - the explanation is JSON that must match a schema;
  - every driver must name a real model input, with the sign and size of its SHAP value;
  - every number in every sentence must come from the model output;
  - diagnosis/instruction language and false claims about the inputs or the scan are removed.
Removed items are shown to the doctor. If the LLM fails, a template built from the numbers is shown.

Backend: any OpenAI-compatible endpoint, by default a local LM Studio server, so no patient text
leaves the clinic network and no API key is needed.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

BASE_URL = os.getenv("FIRSTLOOK_LLM_BASE_URL", "http://localhost:1234/v1")
MODEL = os.getenv("FIRSTLOOK_LLM_MODEL", "local-model")
API_KEY = os.getenv("FIRSTLOOK_LLM_API_KEY", "lm-studio")   # LM Studio ignores the key
TIMEOUT_S = float(os.getenv("FIRSTLOOK_LLM_TIMEOUT", "180"))
MAX_TOKENS = int(os.getenv("FIRSTLOOK_LLM_MAX_TOKENS", "2000"))
# Reasoning models (e.g. Qwen3.x) think for thousands of tokens before answering. This task is
# extraction, not reasoning, so thinking is switched off unless FIRSTLOOK_LLM_THINK=1.
THINK = os.getenv("FIRSTLOOK_LLM_THINK", "0") == "1"
# Some servers/models ignore both switches and think anyway; then all MAX_TOKENS go to reasoning_content and
# the answer is empty. One retry gets this larger budget (about 2-3 minutes at 40 tokens/s).
RETRY_TOKENS = int(os.getenv("FIRSTLOOK_LLM_RETRY_TOKENS", "7000"))


class ThinkingOverrun(RuntimeError):
    """The model spent the whole token budget thinking and never wrote an answer."""


def _answer_text(resp) -> str:
    """The answer part of a completion. reasoning_content is used only if the server finished normally
    (some servers put the whole answer there); a cut-off reasoning trace is never treated as the answer."""
    ch = resp.choices[0]
    text = (ch.message.content or "").strip()
    if not text:
        reasoning = getattr(ch.message, "reasoning_content", "") or ""
        if ch.finish_reason == "length":
            raise ThinkingOverrun(f"thinking used all {resp.usage.completion_tokens if resp.usage else '?'} tokens")
        text = reasoning
    return text


def _create(client, **kwargs):
    """chat.completions.create with one retry at RETRY_TOKENS when the model only thought."""
    try:
        return _answer_text(client.chat.completions.create(**kwargs))
    except ThinkingOverrun:
        if kwargs.get("max_tokens", 0) >= RETRY_TOKENS:
            raise
        kwargs["max_tokens"] = RETRY_TOKENS
        return _answer_text(client.chat.completions.create(**kwargs))


def failure_reason(source: str) -> str:
    """Plain-language reason for a non-live explanation source such as 'template (APIConnectionError)'."""
    if "ThinkingOverrun" in source:
        return ("the local LLM answered but spent its whole token budget thinking and wrote no explanation "
                "(turn thinking off for this model in the LLM server, or raise FIRSTLOOK_LLM_RETRY_TOKENS)")
    if any(k in source for k in ("APIConnectionError", "APITimeoutError", "ConnectError", "Timeout")):
        return "the local LLM could not be reached or did not answer in time"
    return "the local LLM's answer could not be used (it failed the format or number checks)"
PROMPT_VERSION = "v5"   # v5: offline replay removed; claims that the model cannot see the scan are removed


def _extract_json(text: str) -> dict:
    """Parse the reply: drop <think> blocks and code fences, then take the outermost {...}."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


EXPLAIN_PROMPT = """You explain the output of a trained model to a memory-clinic doctor.
The model (a 3D CNN on the T1 MRI plus six clinical inputs; Global CDR is withheld) outputs a probability of
Alzheimer's disease: it was trained to separate patients with Alzheimer's disease from those without
(cognitively normal, subjective complaints, MCI). You receive its numbers; you do not see the scan.
Rules:
- Use only the numbers given. Do not invent findings, scores or image features.
- Do not state a diagnosis, do not recommend treatment, do not tell the doctor what to decide.
- summary: one or two plain sentences on the estimate and how sure the model is.
- key_drivers: the clinical inputs that moved the estimate most (at most 4). For each give the input name
  exactly as listed, direction "raises" or "lowers", effect = its contribution number copied exactly,
  and one short sentence.
- uncertainty_note: one sentence on how sure the model is (fold spread, distance from 0.5).
- limitations: at most 3 short sentences (e.g. model inputs that were filled in, what the model cannot see).
- Fields marked "(not a model input)" did not affect the estimate in any way: never say they were filled in,
  used, or changed the result. The highlighted-region map is shown to the doctor separately; you cannot see it.
- The "cohort-average clinical values" estimate is a reference point (same MRI, average clinical values),
  not the average risk of the cohort.
Answer with JSON only, matching the schema."""

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_drivers": {"type": "array", "items": {
            "type": "object",
            "properties": {"input": {"type": "string"},
                           "direction": {"type": "string", "enum": ["raises", "lowers"]},
                           "effect": {"type": "number"},
                           "text": {"type": "string"}},
            "required": ["input", "direction", "effect", "text"]}},
        "uncertainty_note": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_drivers", "uncertainty_note", "limitations"],
}

# Explanation-specific role check: the explanation may talk about a probability of Alzheimer's disease,
# but it must not turn it into a diagnosis or an instruction.
EXPLAIN_VERDICT = re.compile(
    r"\b(diagnos\w*|(?:has|have|is suffering from) alzheimer|will (?:convert|develop|progress|decline)|"
    r"definitely|certainly|confirm\w*|rules? out|ruled out|recommend\w*|should (?:start|prescribe|treat|refer)|"
    r"prescribe\w*)\b", re.IGNORECASE)
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")
SMALL_WORDS = {0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}     # "five models", "0.5 line", ...


@dataclass
class Explanation:
    summary: str
    drivers: list[dict]
    uncertainty: str
    limitations: list[str]
    removed: list[dict] = field(default_factory=list)
    source: str = "live"
    raw: dict | None = None


def _allowed_numbers(facts: str) -> set[float]:
    vals = set(SMALL_WORDS)
    for m in NUMBER.findall(facts):
        v = float(m.rstrip("%"))
        vals |= {round(v, 4), round(abs(v), 4)}
    return vals


def _number_ok(tok: str, allowed: set[float]) -> bool:
    pct = tok.endswith("%")
    v = abs(float(tok.rstrip("%")))
    cands = [v / 100] if pct else [v]      # a bare number must match as written (no unit guessing)
    return any(abs(c - a) <= (0.006 if c < 1.5 else 0.051) for c in cands for a in allowed)


def _sentence_problem(text: str, allowed: set[float]) -> str | None:
    if EXPLAIN_VERDICT.search(text):
        return "diagnosis or instruction language"
    if SCAN_DENIED.search(text):
        return "says the model does not see the scan (it does: the scan is the model's main input)"
    m = NON_INPUT.search(text)
    if m and (FILLED_WORDS.search(text) or (USED_WORDS.search(text) and not NEGATED.search(text))):
        return f"treats {m.group(0)} as a model input (the model does not use it)"
    bad = [t for t in NUMBER.findall(text) if not _number_ok(t, allowed)]
    if bad:
        return f"number(s) not in the model output: {', '.join(bad)}"
    return None


# The LLM cannot see the scan; the model (a 3D CNN) can. LLMs tend to mix the two up.
SCAN_DENIED = re.compile(r"\bmodel\b[^.]{0,40}\b(?:cannot|can't|does not|doesn't|did not|is unable to)\s+"
                         r"(?:directly\s+)?(?:see|view|access|analy[sz]e|use|read|look at)\b[^.]{0,30}\b(?:MRI|scan|image)", re.I)

# Fields shown to the doctor but never given to the model, and wording that claims the model used them.
NON_INPUT = re.compile(r"\bNPI-?Q\b|\bweight\b", re.I)
FILLED_WORDS = re.compile(r"fill|imput|cohort averag|replaced", re.I)      # never true for a non-input
NEGATED = re.compile(r"\b(?:not|never|n't)\s+(?:\w+\s+){0,2}(?:used|us(?:e|ing)|influenc\w*|affect\w*|"
                     r"contribut\w*|chang\w*|an? (?:model )?input|part of)", re.I)
USED_WORDS = re.compile(r"averag|affect|influenc|contribut|raise|lower|used|accuracy|chang", re.I)


SHAP_FIELD = {"Sex": "sex", "APOE4": "apoe", "Age": "age", "MMSE": "mmse", "FAQ": "faq", "GDS": "gds"}


def _input_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _input_names(shap: dict) -> dict:
    """Every name the LLM may reasonably use for a model input -> the SHAP key: the key itself, the label on
    the chart and the label in the patient record (e.g. 'FAQ functional score (0-30, higher = worse)')."""
    from firstlook.backend import FIELDS, SHAP_LABEL
    names = {}
    for k in shap:
        for alias in (k, SHAP_LABEL.get(k, k), FIELDS.get(SHAP_FIELD.get(k, ""), k)):
            names[_input_key(alias)] = k
            names[_input_key(re.sub(r"\(.*?\)", "", alias))] = k   # label without its bracketed range
    return names


def _resolve_input(given: str, names: dict):
    n = _input_key(given)
    if n in names:
        return names[n]
    n = _input_key(re.sub(r"\(.*?\)", "", given))
    return names.get(n)


def validate_explanation(raw: dict, pred, facts: str) -> Explanation:
    """The model proposes, the system disposes: every sentence is checked against the numbers it was
    given; drivers must name a real input, carry the right sign and the right size."""
    for key in EXPLAIN_SCHEMA["required"]:
        if key not in raw:
            raise ValueError(f"missing key: {key}")
    allowed = _allowed_numbers(facts)
    removed = []

    def keep(text, where):
        if not isinstance(text, str) or not text.strip():
            return None
        why = _sentence_problem(text, allowed)
        if why:
            removed.append({"item": text, "where": where, "reason": why})
            return None
        return text.strip()

    names = _input_names(pred.shap)
    drivers = []
    for d in raw.get("key_drivers", [])[:4]:
        if not isinstance(d, dict) or not all(k in d for k in ("input", "direction", "effect", "text")):
            removed.append({"item": d, "where": "driver", "reason": "malformed"}); continue
        name = _resolve_input(str(d["input"]), names)
        if name is None:
            removed.append({"item": d, "where": "driver", "reason": f"not a model input ({d['input']})"}); continue
        true = pred.shap[name]
        if abs(true) >= 0.002 and (d["direction"] == "raises") != (true > 0):
            removed.append({"item": d, "where": "driver", "reason": f"wrong direction (SHAP {true:+.3f})"}); continue
        try:
            if abs(abs(float(d["effect"])) - abs(true)) > 0.006:
                removed.append({"item": d, "where": "driver",
                                "reason": f"wrong size ({d['effect']} vs SHAP {true:+.3f})"}); continue
        except (TypeError, ValueError):
            removed.append({"item": d, "where": "driver", "reason": "effect is not a number"}); continue
        text = keep(d["text"], "driver")
        if text is None:
            continue
        drivers.append({"input": name, "value": true, "text": text})
    return Explanation(summary=keep(raw.get("summary"), "summary") or "",
                       drivers=drivers,
                       uncertainty=keep(raw.get("uncertainty_note"), "uncertainty") or "",
                       limitations=[t for t in (keep(x, "limitation") for x in raw.get("limitations", [])[:3]) if t],
                       removed=removed, raw=raw)


def template_explanation(pred) -> Explanation:
    """Deterministic fallback when no LLM is available (built only from the numbers)."""
    ranked = sorted(pred.shap.items(), key=lambda kv: -abs(kv[1]))
    drivers = [{"input": k, "value": v, "text": f"{'Raised' if v > 0 else 'Lowered'} the estimate by {abs(v):.3f}."}
               for k, v in ranked if abs(v) >= 0.005][:4]
    unsure = pred.spread >= 0.25 or 0.35 <= pred.held_out_prob <= 0.65
    return Explanation(
        summary=f"The model's estimate is {pred.held_out_prob:.2f}.",
        drivers=drivers,
        uncertainty=("The estimate is close to the decision line or the models disagree, so the model is unsure."
                     if unsure else f"The five models agree within {pred.spread:.2f}."),
        limitations=[] if drivers else ["No single clinical input moved the estimate by 0.005 or more; "
                                        "most of it comes from the MRI."],
        source="template")


def template_text(pred) -> str:
    e = template_explanation(pred)
    return " ".join([e.summary] + [f"{d['input']}: {d['text']}" for d in e.drivers] + [e.uncertainty])


def explain_result(patient_lines: str, pred) -> Explanation:
    """Structured, validated explanation. If the LLM is unreachable or its JSON unusable: a template
    built from the numbers, labelled with the reason."""
    facts = result_facts(patient_lines, pred)
    try:
        from openai import BadRequestError, OpenAI
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT_S, max_retries=0)
        kwargs = dict(model=MODEL, temperature=0.2, max_tokens=MAX_TOKENS,
                      messages=[{"role": "system", "content": EXPLAIN_PROMPT},
                                {"role": "user", "content": facts + "\n\nReturn the JSON now."
                                 + ("" if THINK else " /no_think")}])
        if not THINK:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            text = _create(client, response_format={"type": "json_schema", "json_schema": {
                "name": "explanation", "schema": EXPLAIN_SCHEMA}}, **kwargs)
        except BadRequestError:
            text = _create(client, **kwargs)
        raw = _extract_json(text)
        exp = validate_explanation(raw, pred, facts)
        exp.source = "live"
        return exp
    except Exception as exc:
        exp = template_explanation(pred)
        exp.source = f"template ({type(exc).__name__})"
        return exp


CHAT_PROMPT = """You are the FirstLook assistant inside a memory-clinic review tool. A doctor is reviewing one
patient (identified only by an image ID). Grounding rules:
- Answer only from the context below (the patient record and, if present, the model's numbers).
  If something is not in the context, say it is not recorded. Never invent values, findings or studies.
- You cannot see the MRI. Do not describe image findings.
- Do not diagnose and do not tell the doctor what to decide.
- Do not simply agree with the doctor. If the doctor states something that contradicts the record or the
  model's numbers, say so politely and point to the value. If the model is unsure, say so even if the
  doctor sounds certain.
- Keep answers short (at most 5 sentences)."""
CHAT_BEFORE = ("The doctor has NOT yet recorded their own judgement. The model result is hidden on purpose. "
               "Do not estimate, hint at or discuss the patient's risk, likely diagnosis or the model's output; "
               "if asked, say the AI analysis becomes available after they record their judgement "
               "or press 'View AI analysis'.")


def _complete(messages: list[dict], max_tokens: int = 700) -> str:
    """Plain-text completion against the configured OpenAI-compatible server."""
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT_S, max_retries=0)
    msgs = [dict(m) for m in messages]
    if not THINK:
        msgs[-1]["content"] = msgs[-1]["content"] + " /no_think"
    kwargs = dict(model=MODEL, temperature=0.3, max_tokens=max_tokens, messages=msgs)
    if not THINK:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    text = _create(client, **kwargs)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"/no_think", "", text).strip()
    if not text:
        raise ValueError("empty reply")
    return text


def result_facts(patient_lines: str, pred) -> str:
    """The numbers the LLM may use, as text. pred is a backend.Prediction."""
    folds = ", ".join(f"fold {f['fold']}: {f['prob']:.2f}" + (" (held out)" if f["held_out"] else "")
                      for f in pred.folds)
    shap = "\n".join(f"  {k}: {v:+.3f}" for k, v in sorted(pred.shap.items(), key=lambda kv: -abs(kv[1])))
    base = f"\nEstimate with this MRI but cohort-average clinical values (reference point): {pred.shap_base:.2f}" \
        if pred.shap_base is not None else ""
    return (f"Patient record:\n{patient_lines}\n\n"
            f"MRI usable by the model: {'yes' if pred.mri_ok else 'NO (blank volume used)'}\n"
            + (f"Main estimate (the model that did not see this patient in training): {pred.held_out_prob:.2f}\n"
               if getattr(pred, "estimate_mode", "held_out") == "held_out" else
               f"Main estimate (mean of five models; the uploaded scan is new to all of them): {pred.held_out_prob:.2f}\n")
            +
            f"Distance of the main estimate from 0.5: {abs(pred.held_out_prob - 0.5):.2f}\n"
            f"All five fold models: {folds}; spread {pred.spread:.2f}{base}\n"
            f"Contribution of each clinical input (Shapley values, probability units; + raises the estimate):\n{shap}")


def chat(history: list[dict], patient_lines: str, pred=None) -> str:
    """history: [{"role": "user"|"assistant", "content": str}]. pred=None before the reveal."""
    context = (result_facts(patient_lines, pred) if pred is not None
               else f"Patient record:\n{patient_lines}\n\n{CHAT_BEFORE}")
    msgs = [{"role": "system", "content": CHAT_PROMPT + "\n\nContext:\n" + context}] + history[-10:]
    return _complete(msgs, max_tokens=600)
