"""Presentation helpers: clinical-style header, stepper and cards (HTML/CSS only, no logic)."""
from __future__ import annotations

import html

CSS = """
<style>
  [data-testid="stDecoration"], [data-testid="stAppDeployButton"], #MainMenu, footer {display:none !important;}
  header[data-testid="stHeader"] {background: transparent;}
  .block-container {padding-top: 1.2rem; max-width: 1350px;}
  html, body, [class*="css"] {font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;}
  .fl-bar {background:#1f3a5f; color:#fff; border-radius:8px; padding:10px 18px; display:flex;
           align-items:center; gap:28px; flex-wrap:wrap; margin-bottom:10px;}
  .fl-brand {font-weight:700; font-size:1.05rem; letter-spacing:.3px;}
  .fl-brand small {display:block; font-weight:400; font-size:.72rem; color:#b8c7da;}
  .fl-field {font-size:.78rem; color:#b8c7da; line-height:1.2;}
  .fl-field b {display:block; color:#fff; font-size:.95rem; font-weight:600;}
  .fl-badge {margin-left:auto; background:#f2b134; color:#1f3a5f; font-weight:700; font-size:.72rem;
             padding:4px 10px; border-radius:12px;}
  .fl-cond {background:#2e6f9e; font-size:.72rem; padding:4px 10px; border-radius:12px;}
  .fl-steps {display:flex; gap:6px; margin:6px 0 14px 0;}
  .fl-step {flex:1; text-align:center; padding:7px 4px; border-radius:6px; font-size:.82rem;
            background:#eef2f6; color:#6b7c93; border:1px solid #dde4ec;}
  .fl-step.done {background:#e3efe6; color:#2f6b3a; border-color:#bcd9c2;}
  .fl-step.now {background:#1f3a5f; color:#fff; border-color:#1f3a5f; font-weight:600;}
  .fl-sec {font-size:1.02rem; font-weight:700; color:#1f3a5f; border-bottom:2px solid #1f3a5f;
           padding-bottom:3px; margin:14px 0 8px 0;}
  .fl-chip {display:inline-block; padding:2px 9px; border-radius:10px; font-size:.75rem; font-weight:600;}
  .fl-ok {background:#e3efe6; color:#2f6b3a;} .fl-bad {background:#fbe3e4; color:#a3262b;}
  .fl-note {font-size:.75rem; color:#6b7c93;}
  .fl-verdict {border-radius:8px; padding:14px 18px; font-size:1.15rem; font-weight:700; margin-bottom:8px;}
  .fl-verdict.hi {background:#fbe3e4; color:#a3262b; border-left:6px solid #c0392b;}
  .fl-verdict.lo {background:#e3efe6; color:#2f6b3a; border-left:6px solid #2f8a4a;}
  .fl-verdict small {display:block; font-weight:400; font-size:.8rem; color:#555;}
  .fl-verdict.mid {background:#fff4dc; color:#8a5a00; border-left:6px solid #e0a020;}
  .fl-agree {background:#e3efe6; border-left:6px solid #2f8a4a; border-radius:8px; padding:10px 16px; margin:6px 0;}
  .fl-differ {background:#eef3fa; border-left:6px solid #2e6f9e; border-radius:8px; padding:10px 16px; margin:6px 0;}
  /* floating assistant, bottom-right */
  .st-key-fl_chat {position:fixed; bottom:22px; right:26px; z-index:999; width:auto !important;}
  .st-key-fl_chat button {border-radius:24px; background:#1f3a5f; color:#fff; padding:8px 18px;
                          box-shadow:0 4px 14px rgba(0,0,0,.25);}
  .st-key-fl_chat button p {color:#fff; font-weight:600;}
  [data-testid="stPopoverBody"] {width:440px; max-height:70vh; overflow-y:auto;}
</style>
"""

STEP_NAMES = ["1 Patient & MRI", "2 Your judgement", "3 AI analysis", "4 Final decision"]


def stepper(active: set[int]) -> str:
    """Highlight the steps visible on the current screen; earlier steps are ticked."""
    first = min(active)
    out = []
    for i, n in enumerate(STEP_NAMES, start=1):
        cls = "done" if i < first else "now" if i in active else ""
        out.append(f'<div class="fl-step {cls}">{"✓ " if i < first else ""}{n}</div>')
    return f'<div class="fl-steps">{"".join(out)}</div>'


def section(title: str) -> str:
    return f'<div class="fl-sec">{html.escape(title)}</div>'


def mri_chip(ok: bool, text: str) -> str:
    return f'<span class="fl-chip {"fl-ok" if ok else "fl-bad"}">{html.escape(text)}</span>'


def patient_header(image_id: str, age, sex, cond_label: str) -> str:
    e = html.escape
    question = "Alzheimer's disease assessment"
    fields = [("Image ID", image_id), ("Patient", f"{age if age is not None else '?'} y · {sex}"),
              ("Clinic", "Memory clinic review"), ("Question", question)]
    cells = "".join(f'<div class="fl-field">{e(k)}<b>{e(str(v))}</b></div>' for k, v in fields)
    return (f'<div class="fl-bar"><div class="fl-brand">FirstLook<small>Clinical decision support</small></div>'
            f'{cells}</div>')


def ai_card(prob: float, fold: int | None, band: str, ensemble: bool = False) -> str:
    cls = {"higher": "hi", "lower": "lo"}.get(band, "mid")
    word = {"higher": "higher concern", "lower": "lower concern", "uncertain": "model unsure"}[band]
    src = ("average of five models (uploaded scan, new to all of them)" if ensemble else
           "from the model that did not see this patient in training" if fold else "")
    return (f'<div class="fl-verdict {cls}">AI estimate: {prob:.2f} → {word}'
            f'<small>{"Estimated probability of Alzheimer\'s disease"}'
            f'{" · " + html.escape(src) if src else ""}</small></div>')


LANDING_CSS = """
<style>
  .stApp {background: url("data:image/jpeg;base64,%s") center/cover no-repeat fixed;}
  .block-container {padding-top: 9vh;}
  .fl-hero {text-align:center; color:#fff; margin-bottom:22px;}
  .fl-hero h1 {font-size:2.6rem; font-weight:700; letter-spacing:.5px; margin:0; color:#fff;}
  .fl-hero p {color:#b8c7da; font-size:1rem; margin:6px 0 0 0;}
  .st-key-fl_search {background:rgba(255,255,255,.97); border-radius:14px; padding:26px 28px 18px 28px;
                     box-shadow:0 18px 50px rgba(0,0,0,.35);}
  .fl-foot {text-align:center; color:#8ea3bd; font-size:.75rem; margin-top:18px;}
  [data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] * {color:#dce6f2 !important; opacity:1 !important;}
  [data-testid="stStatusWidget"] {color:#dce6f2;}
</style>
"""


def landing_css(bg_b64: str) -> str:
    return LANDING_CSS % bg_b64


def hero() -> str:
    sub = "Memory clinic · decision support"
    return f'<div class="fl-hero"><h1>FirstLook</h1><p>{html.escape(sub)}</p></div>'


def loader(text: str) -> str:
    """Full-screen loading overlay: pulsing scan rings (pure CSS, no external assets)."""
    return f"""
<style>
  .fl-load {{position:fixed; inset:0; z-index:9999; background:rgba(15,32,56,.82); display:flex;
             flex-direction:column; align-items:center; justify-content:center; color:#fff;}}
  .fl-ring {{position:relative; width:120px; height:120px;}}
  .fl-ring span {{position:absolute; inset:0; border:3px solid #6fb3ff; border-radius:50%;
                  animation: fl-pulse 1.8s ease-out infinite; opacity:0;}}
  .fl-ring span:nth-child(2) {{animation-delay:.6s;}}
  .fl-ring span:nth-child(3) {{animation-delay:1.2s;}}
  .fl-ring b {{position:absolute; left:50%; top:50%; width:34px; height:34px; margin:-17px 0 0 -17px;
               border-radius:50%; background:#6fb3ff; box-shadow:0 0 24px #6fb3ff;}}
  .fl-scan {{width:160px; height:3px; margin-top:26px; background:linear-gradient(90deg,transparent,#6fb3ff,transparent);
             background-size:200% 100%; animation: fl-sweep 1.2s linear infinite;}}
  .fl-load p {{margin-top:14px; font-size:1.05rem; letter-spacing:.3px;}}
  @keyframes fl-pulse {{0% {{transform:scale(.3); opacity:.9;}} 100% {{transform:scale(1.25); opacity:0;}}}}
  @keyframes fl-sweep {{0% {{background-position:200% 0;}} 100% {{background-position:-200% 0;}}}}
</style>
<div class="fl-load"><div class="fl-ring"><span></span><span></span><span></span><b></b></div>
<div class="fl-scan"></div><p>{html.escape(text)}</p></div>"""
