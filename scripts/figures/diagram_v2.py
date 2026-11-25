"""FirstLook system diagram (Figure 1), doctor-first version. English and Chinese labels.
Run:  python diagram_v2.py OUTDIR   ->  system_diagram_v2_en.svg/png, system_diagram_v2_zh.svg/png
Hand-laid SVG: box and arrow positions are set below; labels come from L[lang].
"""
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
W, H = 1420, 900
STY = {'impl': ('#e2f0d9', '#3a7d2c', ''), 'ext': ('#e4dfec', '#7030a0', ''),
       'sim': ('#fff2cc', '#bf8f00', '7 4'), 'gate': ('#fce4d6', '#c55a11', '')}

L = {
 'en': dict(
  font='Helvetica,Arial,sans-serif',
  z_live='Live inference · GPU PC on the local network',
  z_core='FirstLook app (Python)', z_ui='Streamlit UI · Condition B (doctor-first)',
  db=['Patient database (SQLite)', 'from the ADNI export', 'no diagnosis group stored'],
  mri=['T1 scans', 'stored, or uploaded on the search page'],
  mod=['Five fold models', '3D CNN + FiLM'],
  inf=['Inference service (FastAPI)', 'held-out fold or five-model mean', 'exact SHAP · Grad-CAM'],
  llm=['Local LLM (LM Studio)', 'any OpenAI-compatible model'],
  chk=['Input checks (rules)', 'record only: missing / assumed inputs,', 'range, GDS, failed scan'],
  unc=['Uncertainty checks (rules)', 'scores only: fold spread, near 0.5'],
  val=['Validators', 'explanation: input, sign, size, numbers,', 'no diagnosis · no false input/scan claims'],
  gate=['Assistant context gate', 'before Step 2: record only', 'after Step 2: + model numbers'],
  s0=['Search: image ID (+ optional scan)'], s1=['1  Record, own MRI, input alerts'],
  s2=['2  Own read, or name the', 'uncertainty → View AI'], s3=['3  AI analysis: agree → folded;', 'differ / unsure → opened'],
  s4=['4  Lower / Uncertain-refer / Higher', '(reason if the first read changes)'],
  log=['Decision log (JSONL)'], A=['Condition A baseline', 'verdict + SHAP + map first'],
  chat=['Ask the AI assistant (floating)'], doc='Clinician',
  e_rec='record + scan', e_alert='input alerts', e_sc='scores · SHAP · map', e_after='only after Step 2',
  e_json='proposed JSON', e_expl='checked explanation', e_qa='question / answer', e_ask='asks',
  e_judge='judges', e_block='block (MRI fail)', e_A='same backend', e_dec='decides', e_samelog='same log',
  leg=['Implemented', 'External live call', 'Isolation gate'], leg_p='blue = person action'),
 'zh': dict(
  font='Noto Sans CJK SC,Source Han Sans SC,Microsoft YaHei,sans-serif',
  z_live='实机推理 · 局域网内的 GPU 电脑',
  z_core='FirstLook 应用（Python）', z_ui='Streamlit 界面 · 条件 B（医生先判断）',
  db=['患者数据库（SQLite）', '由 ADNI 导出表生成', '不存储诊断分组'],
  mri=['T1 扫描', '库存，或在首页上传'],
  mod=['五折模型', '3D CNN + FiLM'],
  inf=['推理服务（FastAPI）', '留出折 或 五模型均值', '精确 SHAP · Grad-CAM'],
  llm=['本地 LLM（LM Studio）', '任意 OpenAI 兼容模型'],
  chk=['输入检查（规则）', '只看病历：缺失 / 假定输入、', '范围、GDS、扫描读取失败'],
  unc=['不确定性检查（规则）', '只看分数：折间分歧、接近 0.5'],
  val=['校验器', '解释：输入名、方向、大小、数字、', '不得诊断 · 不得误称输入或扫描'],
  gate=['助手上下文隔离', '第 2 步前：只有病历', '第 2 步后：加上模型数值'],
  s0=['查找：影像编号（可上传扫描）'], s1=['1  病历、本人 MRI、输入提醒'],
  s2=['2  先自己判断，或说明', '不确定在哪 → 查看 AI'], s3=['3  AI 分析：一致 → 折叠；', '不一致 / 不确定 → 展开'],
  s4=['4  低 / 不确定-转诊 / 高', '（改判需填理由）'],
  log=['决策日志（JSONL）'], A=['条件 A 基线', '先给结论 + SHAP + 热图'],
  chat=['AI 助手浮窗'], doc='医生',
  e_rec='病历 + 扫描', e_alert='输入提醒', e_sc='分数 · SHAP · 热图', e_after='仅在第 2 步之后',
  e_json='LLM 提议的 JSON', e_expl='校验后的解释', e_qa='提问 / 回答', e_ask='提问',
  e_judge='判断', e_block='扫描失败即阻断', e_A='同一后端', e_dec='决定', e_samelog='同一日志',
  leg=['已实现', '外部实时调用', '隔离关卡'], leg_p='蓝色 = 人的操作'),
}

BW = 260
nodes = {  # id: (x, y, h, style)
 'db': (45, 60, 74, 'impl'), 'mri': (45, 150, 58, 'impl'), 'mod': (45, 224, 74, 'impl'),
 'inf': (45, 330, 74, 'impl'), 'llm': (45, 500, 58, 'ext'),
 'chk': (430, 110, 74, 'impl'), 'unc': (430, 300, 58, 'impl'), 'val': (430, 450, 74, 'impl'),
 'gate': (430, 610, 74, 'gate'),
 's0': (830, 40, 50, 'impl'), 's1': (830, 120, 50, 'impl'), 's2': (830, 200, 62, 'impl'),
 's3': (830, 300, 62, 'impl'), 's4': (830, 412, 62, 'impl'), 'log': (830, 520, 50, 'impl'),
 'A': (830, 610, 62, 'impl'), 'chat': (830, 720, 50, 'impl'),
}


def m(n, side, f=0.5):
    x, y, h, _ = nodes[n]
    return {'l': (x, y + h * f), 'r': (x + BW, y + h * f), 't': (x + BW * f, y), 'b': (x + BW * f, y + h)}[side]


def build(lang):
    t = L[lang]
    E = [  # points, label key, label position, style, two-headed
     ([(30, 97), (30, 367), m('inf', 'l')], None, None, 'impl', False),
     ([m('inf', 'r', .3), (380, 352), (380, 147), m('chk', 'l')], 'e_rec', (386, 250), 'impl', False),
     ([m('chk', 'r'), (780, 147), (780, 145), m('s1', 'l')], 'e_alert', (700, 138), 'impl', False),
     ([m('inf', 'r', .7), (395, 382), (395, 329), m('unc', 'l')], 'e_sc', (402, 418), 'impl', False),
     ([m('unc', 'r', .35), m('s3', 'l', .55)], 'e_after', (700, 312), 'gate', False),
     ([m('llm', 'r', .35), (400, 520), (400, 470), m('val', 'l', .27)], 'e_json', (316, 512), 'ext', False),
     ([m('val', 'r', .35), (790, 476), (790, 350), m('s3', 'l', .8)], 'e_expl', (694, 497), 'impl', False),
     ([m('gate', 'l'), (330, 647), (330, 540), m('llm', 'r', .7)], 'e_qa', (336, 600), 'ext', True),
     ([m('chat', 'l'), (760, 745), (760, 660), m('gate', 'r', .7)], 'e_qa', (766, 700), 'gate', True),
     ([m('chk', 'r', .8), (765, 169), (765, 312), m('s3', 'l', .2)], 'e_block', (636, 250), 'impl', False),
     ([m('inf', 'r', .95), (360, 400), (360, 575), (800, 575), (800, 641), m('A', 'l')], 'e_A', (480, 568), 'impl', False),
     ([m('s0', 'b'), m('s1', 't')], None, None, 'ui', False), ([m('s1', 'b'), m('s2', 't')], None, None, 'ui', False),
     ([m('s2', 'b'), m('s3', 't')], None, None, 'ui', False), ([m('s3', 'b'), m('s4', 't')], None, None, 'ui', False),
     ([m('s4', 'b'), m('log', 't')], None, None, 'impl', False),
     ([m('A', 't'), m('log', 'b')], 'e_samelog', (970, 600), 'impl', False),
     ([(1270, 330), (1200, 330), (1200, 231), m('s2', 'r')], 'e_judge', (1205, 222), 'person', False),
     ([(1270, 330), (1200, 330), (1200, 443), m('s4', 'r')], 'e_dec', (1205, 465), 'person', False),
     ([(1270, 330), (1200, 330), (1200, 745), m('chat', 'r')], 'e_ask', (1205, 738), 'person', False),
    ]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{t["font"]}">',
         '<defs>' + ''.join(
             f'<marker id="{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M0,0L10,5L0,10z" fill="{c}"/></marker>' for k, c in
             [('ar', '#444'), ('arb', '#2e75b6'), ('arp', '#7030a0'), ('aro', '#c55a11'), ('ary', '#bf8f00')]) + '</defs>',
         '<rect width="100%" height="100%" fill="white"/>']
    for x, y, w, h, key, dash in [(20, 20, 330, 600, 'z_live', ''),
                                  (410, 85, 330, 717, 'z_core', ''), (810, 18, 300, 784, 'z_ui', '')]:
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#fafafa" stroke="#bbbbbb" stroke-dasharray="{dash}"/>'
                 f'<text x="{x + 10}" y="{y + h - 9}" font-size="13" fill="#666" font-weight="bold">{t[key]}</text>')
    col = {'person': ('#2e75b6', 'arb'), 'ext': ('#7030a0', 'arp'), 'gate': ('#c55a11', 'aro'), 'sim': ('#bf8f00', 'ary')}
    for pts, lab, lp, st, two in E:
        c, mk = col.get(st, ('#444', 'ar'))
        dash = '7 4' if st == 'sim' else ''
        d = 'M' + ' L'.join(f'{x},{y}' for x, y in pts)
        end = '' if pts[0] == (30, 97) and False else f'marker-end="url(#{mk})"'
        start = f'marker-start="url(#{mk})"' if two else ''
        o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{2.2 if st in ("ui", "person") else 1.7}" '
                 f'stroke-dasharray="{dash}" {end} {start}/>')
        if lab:
            o.append(f'<text x="{lp[0]}" y="{lp[1]}" font-size="12.5" fill="{c}" font-style="italic">{t[lab]}</text>')
    for k in ('db', 'mri', 'mod'):        # stores feed the service (bus on the left)
        x, y, h, _ = nodes[k]
        o.append(f'<path d="M{x},{y + h / 2} L30,{y + h / 2}" stroke="#444" stroke-width="1.7"/>')
    for k, (x, y, h, st) in nodes.items():
        f, s, dash = STY[st]
        o.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{h}" rx="9" fill="{f}" stroke="{s}" stroke-width="2" stroke-dasharray="{dash}"/>')
        lines = t[k]
        for i, ln in enumerate(lines):
            yy = y + h / 2 + (i - (len(lines) - 1) / 2) * 17 + 5
            o.append(f'<text x="{x + BW / 2}" y="{yy}" font-size="{14 if i == 0 else 12.3}" text-anchor="middle" '
                     f'font-weight="{"bold" if i == 0 else "normal"}" fill="#111">{ln}</text>')
    o.append(f'<circle cx="1320" cy="330" r="50" fill="#ddebf7" stroke="#2e75b6" stroke-width="2"/>'
             f'<text x="1320" y="335" font-size="15" text-anchor="middle" font-weight="bold">{t["doc"]}</text>')
    lx, ly = 30, 860
    for lab, st in zip(t['leg'], ['impl', 'ext', 'gate']):
        f, s, dash = STY[st]
        o.append(f'<rect x="{lx}" y="{ly}" width="26" height="18" fill="{f}" stroke="{s}" stroke-width="2" stroke-dasharray="{dash}"/>'
                 f'<text x="{lx + 34}" y="{ly + 14}" font-size="13">{lab}</text>')
        lx += 230
    o.append(f'<text x="{lx + 10}" y="{ly + 14}" font-size="13" fill="#2e75b6">{t["leg_p"]}</text></svg>')
    return '\n'.join(o)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in ('en', 'zh'):
        svg = OUT / f'system_diagram_v2_{lang}.svg'
        svg.write_text(build(lang), encoding='utf-8')
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                b = p.chromium.launch(); pg = b.new_page(viewport={'width': W, 'height': H}, device_scale_factor=2)
                pg.goto(svg.resolve().as_uri()); pg.screenshot(path=str(svg.with_suffix('.png'))); b.close()
        except ImportError:
            pass
        print(svg)
