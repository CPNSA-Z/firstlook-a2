"""Model definition and MRI preprocessing, copied from model_training/train.py (the script that
produced the Twostage_PRO checkpoints). Inference only: dropout is inactive in eval() mode."""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn

TARGET_SHAPE = (160, 192, 160)
CLINICAL_NAMES = ["Sex", "APOE4", "Age", "MMSE", "Global CDR", "FAQ", "GDS"]   # verified against idaSearch (exact match, 990/990)


# ---------------------------------------------------------------- preprocessing (train.py)
def get_brain_bounding_box(img_data):
    nz = np.where(img_data > 0)
    if len(nz[0]) == 0:
        return (0, img_data.shape[0]), (0, img_data.shape[1]), (0, img_data.shape[2])
    pad = 2
    return tuple((max(0, int(np.min(a)) - pad), min(img_data.shape[i], int(np.max(a)) + pad))
                 for i, a in enumerate(nz))


def preprocess_mri(img_data, target_shape=TARGET_SHAPE):
    (x0, x1), (y0, y1), (z0, z1) = get_brain_bounding_box(img_data)
    cropped = img_data[x0:x1, y0:y1, z0:z1]
    scale = min(target_shape[i] / cropped.shape[i] for i in range(3))
    resized = ndimage.zoom(cropped, (scale, scale, scale), order=1)
    pads = [target_shape[i] - resized.shape[i] for i in range(3)]
    padded = np.pad(resized, [(p // 2, p - p // 2) for p in pads], mode="constant")
    return (padded - np.mean(padded)) / (np.std(padded) + 1e-8)


# ---------------------------------------------------------------- architecture (train.py)
class CBAM_3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid_ca = nn.Sigmoid()
        self.conv_sa = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_sa = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        avg_pool = nn.AdaptiveAvgPool3d(1)(x).view(b, c)
        max_pool = nn.AdaptiveMaxPool3d(1)(x).view(b, c)
        ca = self.fc2(self.relu(self.fc1(avg_pool))) + self.fc2(self.relu(self.fc1(max_pool)))
        x = x * self.sigmoid_ca(ca).view(b, c, 1, 1, 1)
        sa = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        return x * self.sigmoid_sa(self.conv_sa(sa))


class FiLMFusion(nn.Module):
    """train.py: img * gamma + beta. (run_shap.py used img * (1 + gamma) + beta; see NOTES.)"""
    def __init__(self, img_dim, clin_dim, one_plus_gamma=False):
        super().__init__()
        self.film_gen = nn.Linear(clin_dim, img_dim * 2)
        self.one_plus_gamma = one_plus_gamma

    def forward(self, img_f, clin_f):
        gamma, beta = torch.chunk(self.film_gen(clin_f), 2, dim=1)
        return img_f * ((1 + gamma) if self.one_plus_gamma else gamma) + beta


class SeparableConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch, k, s, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv3d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, pool=False, drop=0.1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, tuple(i // 2 for i in k), bias=False)
        self.bn, self.elu, self.drop = nn.BatchNorm3d(out_ch), nn.ELU(inplace=True), nn.Dropout3d(drop)
        self.pool = nn.MaxPool3d(3, 2) if pool else None

    def forward(self, x):
        x = self.drop(self.elu(self.bn(self.conv(x))))
        return self.pool(x) if self.pool else x


class FCBlock(nn.Module):
    def __init__(self, in_f, out_f, drop=0.1):
        super().__init__()
        self.fc, self.bn = nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f)
        self.elu, self.drop = nn.ELU(inplace=True), nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.elu(self.bn(self.fc(x))))


class TwoStageAlex3D(nn.Module):
    def __init__(self, num_clinical=7, drop=0.1, one_plus_gamma=False):
        super().__init__()
        self.conv1 = ConvBlock(1, 24, (11, 13, 11), 4, True, drop)
        self.conv2 = ConvBlock(24, 48, (5, 6, 5), 1, True, drop)
        self.cbam = CBAM_3D(channels=48)
        self.sep = nn.Sequential(*[
            nn.Sequential(SeparableConv3d(48, 48, (3, 3, 3), padding=1),
                          nn.BatchNorm3d(48), nn.ELU(), nn.Dropout3d(drop))
            for _ in range(3)])
        self.conv5 = ConvBlock(48, 24, (3, 4, 3), 1, True, drop)
        self.conv6 = ConvBlock(24, 8, (3, 4, 3), 1, False, drop)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.img_fc = FCBlock(8, 10, drop)
        self.img_only_out = nn.Linear(10, 1)
        self.clin_fc1 = FCBlock(num_clinical, 32, drop)
        self.clin_fc2 = FCBlock(32, 10, drop)
        self.film_fusion = FiLMFusion(10, 10, one_plus_gamma)
        self.out = nn.Linear(10, 1)

    def image_features(self, mri):
        x = self.conv2(self.conv1(mri))
        x = self.cbam(x)
        x = x + self.sep(x)
        return x

    def forward(self, mri, clin):
        x = self.image_features(mri)
        img_f = self.pool(self.conv6(self.conv5(x))).view(x.size(0), -1)
        fused = self.film_fusion(self.img_fc(img_f), self.clin_fc2(self.clin_fc1(clin)))
        return torch.sigmoid(self.out(fused))


def load_fold(path, device="cpu", one_plus_gamma=False):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TwoStageAlex3D(7, drop=0.1, one_plus_gamma=one_plus_gamma).to(device)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck, strict=True)
    model.eval()
    return model, float(ck.get("best_auc", float("nan")))
