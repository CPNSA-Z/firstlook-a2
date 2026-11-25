"""Explanations for one patient: exact Shapley values for the clinical inputs, smoothed Grad-CAM
for the MRI, and slice rendering.

Clinical Shapley values: the image embedding is fixed and the 7 clinical inputs are switched
between the patient's value and the cohort mean (0 after z-scoring). With 7 inputs all 2^7 = 128
coalitions are evaluated, so the values are exact (no sampling), and they sum to
f(patient) - f(all clinical inputs at the cohort mean).
"""
from __future__ import annotations

import io
import itertools
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402
import torch.nn.functional as F   # noqa: E402

from fl_model import CLINICAL_NAMES, TARGET_SHAPE   # noqa: E402

SMOOTH_SAMPLES = 8      # interpret_image.py used 10
NOISE_STD = 0.05


# ---------------------------------------------------------------- model pieces
@torch.no_grad()
def image_embedding(model, mri):
    x = model.image_features(mri)
    img_f = model.pool(model.conv6(model.conv5(x))).view(x.size(0), -1)
    return model.img_fc(img_f)


@torch.no_grad()
def head(model, img_embed, clin):
    fused = model.film_fusion(img_embed.expand(clin.size(0), -1), model.clin_fc2(model.clin_fc1(clin)))
    return torch.sigmoid(model.out(fused)).squeeze(1)


@torch.no_grad()
def clinical_shapley(model, mri, clin):
    """Exact Shapley values of the 7 clinical inputs (baseline: cohort mean = 0)."""
    n = clin.size(1)
    emb = image_embedding(model, mri)
    masks = torch.tensor(list(itertools.product([0, 1], repeat=n)), dtype=clin.dtype, device=clin.device)
    preds = head(model, emb, masks * clin)                    # 2^n forward passes of the small head
    value = {tuple(m.int().tolist()): p.item() for m, p in zip(masks, preds)}
    phi = np.zeros(n)
    for i in range(n):
        for m, v in value.items():
            if m[i] == 1:
                continue
            s = sum(m)
            w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            with_i = m[:i] + (1,) + m[i + 1:]
            phi[i] += w * (value[with_i] - v)
    base = value[(0,) * n]
    return dict(zip(CLINICAL_NAMES, phi.round(4).tolist())), round(base, 4)


def smooth_gradcam(model, mri, clin, samples=SMOOTH_SAMPLES):
    """Smoothed Grad-CAM at conv5, as in interpret_image.py. Returns a [0,1] map in model space."""
    store = {}
    layer = model.conv5.conv
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o.detach()))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0].detach()))
    cam_sum = np.zeros(TARGET_SHAPE, dtype=np.float32)
    mask = (mri[0, 0] > mri.min() + 1e-3).float().cpu().numpy()
    try:
        for _ in range(samples):
            x = (mri + torch.randn_like(mri) * NOISE_STD).requires_grad_(True)
            model.zero_grad()
            model(x, clin).sum().backward()
            w = store["g"].mean(dim=(2, 3, 4), keepdim=True)
            cam = F.relu((w * store["a"]).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=TARGET_SHAPE, mode="trilinear", align_corners=False)
            cam_sum += cam[0, 0].cpu().numpy() * mask
    finally:
        h1.remove(); h2.remove()
    cam = cam_sum / samples
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


# ---------------------------------------------------------------- rendering
def _slices(vol, at):
    x, y, z = at
    return [np.rot90(vol[x, :, :]), np.rot90(vol[:, y, :]), np.rot90(vol[:, :, z])]


def render_png(vol, overlay=None, at=None, title_suffix=""):
    """Three orthogonal slices (sagittal, coronal, axial) of a 3-D volume, optional heat overlay."""
    v = vol.astype(np.float32)
    brain = v > v.min() + 1e-3
    if brain.any():                       # crop to the brain so the slices fill the panel
        idx = np.where(brain)
        box = tuple(slice(max(0, i.min() - 3), i.max() + 4) for i in idx)
        v = v[box]
        overlay = overlay[box] if overlay is not None else None
        lo, hi = np.percentile(v[v > v.min()], [1, 99])
    else:
        lo, hi = 0.0, 1.0
    v = np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)
    if at is None:   # centre of the brain: same slices for the scan and the attribution view
        at = tuple(s // 2 for s in v.shape)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), facecolor="black")
    for ax, img, name, ov in zip(axes, _slices(v, at), ["Sagittal", "Coronal", "Axial"],
                                 _slices(overlay, at) if overlay is not None else [None] * 3):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        if ov is not None:
            ax.imshow(np.ma.masked_less(ov, 0.15), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(name + title_suffix, color="white", fontsize=9)
        ax.axis("off")
    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="black")
    plt.close(fig)
    return buf.getvalue()
