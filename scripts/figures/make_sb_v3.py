"""Compose the three storyboards from screenshots of a real local run (Chrome, live mode)."""
import sys
from PIL import Image, ImageDraw, ImageFont
SRC = sys.argv[1] if len(sys.argv) > 1 else 'docs/screens/'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'docs/'
font = lambda s: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', s)
fontr = lambda s: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', s)
COL = {'P': (46, 117, 182), 'S': (192, 80, 77), 'AI': (112, 48, 160)}
NOTE = ("P = person acts · S = rule-based system · AI = model or LLM output.  Screens from a real local run: inference "
        "service on a GPU PC, local LLM; image IDs replaced by pseudonyms. All LLM text is the local model's own output.")
W, PAD, HDR, MAXH = 1000, 24, 86, 760

def fit(im):
    s = min((W - 2 * PAD) / im.width, MAXH / im.height)
    return im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)

def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ''
    for w in words:
        if d.textlength(cur + ' ' + w, font=f) > width and cur:
            lines.append(cur); cur = w
        else:
            cur = (cur + ' ' + w).strip()
    return lines + [cur]

def board(title, frames, out):
    cells = [(fit(Image.open(SRC + n + '.png').convert('RGB')), who, lab) for n, who, lab in frames]
    rowh = [max(c[0].height for c in cells[i:i + 2]) + HDR + 2 * PAD for i in (0, 2)]
    tmp = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    note = wrap(tmp, NOTE, fontr(20), 2 * W - 2 * PAD)
    canvas = Image.new('RGB', (2 * W + PAD, 90 + sum(rowh) + 30 * len(note) + PAD), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((PAD, 24), title, fill=(30, 30, 30), font=font(34))
    y = 90
    for r in range(2):
        for c in range(2):
            i = 2 * r + c
            im, who, label = cells[i]
            x = c * W + PAD // 2
            d.rectangle([x, y, x + W - PAD // 2, y + rowh[r] - PAD // 2], outline=(180, 180, 180), width=3)
            d.ellipse([x + 14, y + 14, x + 62, y + 62], fill=(40, 40, 40))
            d.text((x + 30, y + 20), str(i + 1), fill='white', font=font(30))
            tx = x + 76
            for tag in (['AI'] if who == 'AI' else ['AI', 'P'] if who == 'AIP' else list(who)):
                w = d.textlength(tag, font=font(22)) + 20
                d.rounded_rectangle([tx, y + 20, tx + w, y + 54], 8, fill=COL[tag])
                d.text((tx + 10, y + 24), tag, fill='white', font=font(22)); tx += w + 8
            d.text((tx + 6, y + 22), label, fill=(20, 20, 20), font=fontr(25))
            px = x + PAD // 2 + (W - 2 * PAD - im.width) // 2
            canvas.paste(im, (px, y + HDR))
            d.rectangle([px, y + HDR, px + im.width, y + HDR + im.height], outline=(220, 220, 220), width=2)
        y += rowh[r]
    for k, line in enumerate(note):
        d.text((PAD, y + 4 + 30 * k), line, fill=(90, 90, 90), font=fontr(20))
    canvas.save(out); print(out, canvas.size)

board("Storyboard 1 · Doctor first: record and scan, then the doctor, then the model (CASE-001)", [
    ('01_search', 'P', "Search by image ID (scan upload optional)"),
    ('02_step1', 'S', "Step 1: record, patient's own MRI; no AI output"),
    ('03_step2_locked', 'P', "Step 2: own read first; 'View AI' still locked"),
    ('04_agree', 'S', "Agrees with model: short note, details folded")], OUT + 'storyboard1_doctor_first.png')
board("Storyboard 2 · Disagreement: the analysis opens, every LLM sentence is checked (CASE-002)", [
    ('05_differ_step3', 'AI', "Differs: neutral prompt, estimate, folds, SHAP, map"),
    ('06_words', 'AI', "Local LLM explanation; every sentence passed the checks"),
    ('08_final_reason', 'P', "Changes to Higher: reason required; 'Uncertain' offered"),
    ('09_condA', 'AI', "Baseline A, same patient: verdict shown first")], OUT + 'storyboard2_disagreement.png')
board("Storyboard 3 · Guarding the AI: isolation, the uncertainty gate and an uploaded scan", [
    ('11_chat_before', 'AI', "Assistant before the judgement: record only"),
    ('12_gate', 'P', "Not sure: name the uncertainty, then View AI"),
    ('14_chat_after', 'AI', "After the reveal: grounded in the numbers"),
    ('16_upload_step3', 'S', "Uploaded scan: mean of five equal models")], OUT + 'storyboard3_guarding_ai.png')
