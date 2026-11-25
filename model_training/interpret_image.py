import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndimage
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

class InterpretConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_SHAPE = (160, 192, 160)
    
    PROJECT_ROOT = Path("/path/to/MCI_Project")
    CLEAN_DATA_DIR = PROJECT_ROOT / "outputs" / "Clean_Data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "Interpret_Results"
    DATA_PKL_PATH = CLEAN_DATA_DIR / "df_cohort_final.pkl"
    
    MODEL_PATH = PROJECT_ROOT / "outputs" / "Saved_Models" / "Twostage_PRO_fold3_AUC0.9561.pth"
    
    SLICE_COORD = [80, 100, 75] 

    SMOOTH_SAMPLES = 10     
    NOISE_STD = 0.05        

    def __init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

Config = InterpretConfig()

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
        b, c, _, _, _ = x.size()
        avg_pool = nn.AdaptiveAvgPool3d(1)(x).view(b, c)
        max_pool = nn.AdaptiveMaxPool3d(1)(x).view(b, c)
        ca = self.fc2(self.relu(self.fc1(avg_pool))) + self.fc2(self.relu(self.fc1(max_pool)))
        ca = self.sigmoid_ca(ca).view(b, c, 1, 1, 1)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid_sa(self.conv_sa(sa))
        return x * sa

class SeparableConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv3d(in_ch, in_ch, k, s, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv3d(in_ch, out_ch, 1, bias=False)
    def forward(self, x): return self.pointwise(self.depthwise(x))

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, pool=False, drop=0.1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, k, s, tuple(i//2 for i in k), bias=False)
        self.bn, self.elu, self.drop = nn.BatchNorm3d(out_ch), nn.ELU(inplace=True), nn.Dropout3d(drop)
        self.pool = nn.MaxPool3d(3, 2) if pool else None
    def forward(self, x):
        x = self.drop(self.elu(self.bn(self.conv(x))))
        return self.pool(x) if self.pool else x

class FCBlock(nn.Module):
    def __init__(self, in_f, out_f, drop=0.1):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)
        self.bn = nn.BatchNorm1d(out_f)
        self.elu = nn.ELU(inplace=True)
        self.drop = nn.Dropout(drop)
    def forward(self, x): return self.drop(self.elu(self.bn(self.fc(x))))

class FiLMLayer(nn.Module):
    def __init__(self, img_dim, clin_dim):
        super().__init__()
        self.film_gen = nn.Linear(clin_dim, img_dim * 2)
    def forward(self, img_features, clin_features):
        film_params = self.film_gen(clin_features)
        gamma = film_params[:, :img_features.size(1)]
        beta = film_params[:, img_features.size(1):]
        return (img_features * (1 + gamma)) + beta

class TwoStageAlex3D(nn.Module):
    def __init__(self, num_clinical=7, drop=0.0): 
        super().__init__()
        self.conv1 = ConvBlock(1, 24, (11,13,11), 4, True, drop)
        self.conv2 = ConvBlock(24, 48, (5,6,5), 1, True, drop)
        self.cbam = CBAM_3D(channels=48)
        self.sep = nn.Sequential(*[
            nn.Sequential(SeparableConv3d(48,48,(3,3,3),padding=1), nn.BatchNorm3d(48), nn.ELU(), nn.Dropout3d(drop))
            for _ in range(3)
        ])
        self.conv5 = ConvBlock(48, 24, (3,4,3), 1, True, drop)
        self.conv6 = ConvBlock(24, 8, (3,4,3), 1, False, drop) 
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.img_fc = FCBlock(8, 10, drop)
        
        self.clin_fc1 = FCBlock(num_clinical, 32, drop)
        self.clin_fc2 = FCBlock(32, 10, drop)
        self.film_fusion = FiLMLayer(img_dim=10, clin_dim=10)
        self.out = nn.Linear(10, 1)

    def forward(self, mri, clin):
        x = self.conv2(self.conv1(mri))
        x = self.cbam(x)
        x = x + self.sep(x)
        img_f = self.pool(self.conv6(self.conv5(x))).view(x.size(0), -1)
        img_embed = self.img_fc(img_f)
        clin_embed = self.clin_fc2(self.clin_fc1(clin))
        fused = self.film_fusion(img_embed, clin_embed)
        return torch.sigmoid(self.out(fused))

def preprocess_mri(img_data, target_shape):
    non_zero = np.where(img_data > 0)
    if len(non_zero[0]) == 0: return np.zeros(target_shape)
    pad = 2
    x_min, x_max = max(0, np.min(non_zero[0]) - pad), min(img_data.shape[0], np.max(non_zero[0]) + pad)
    y_min, y_max = max(0, np.min(non_zero[1]) - pad), min(img_data.shape[1], np.max(non_zero[1]) + pad)
    z_min, z_max = max(0, np.min(non_zero[2]) - pad), min(img_data.shape[2], np.max(non_zero[2]) + pad)
    cropped = img_data[x_min:x_max, y_min:y_max, z_min:z_max]
    scale = min(target_shape[0]/cropped.shape[0], target_shape[1]/cropped.shape[1], target_shape[2]/cropped.shape[2])
    resized = ndimage.zoom(cropped, (scale, scale, scale), order=1)
    px, py, pz = target_shape[0]-resized.shape[0], target_shape[1]-resized.shape[1], target_shape[2]-resized.shape[2]
    padded = np.pad(resized, ((px//2, px-px//2), (py//2, py-py//2), (pz//2, pz-pz//2)), mode='constant')
    return (padded - np.mean(padded)) / (np.std(padded) + 1e-8)

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): self.activations = output.detach()
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0].detach()

def run_interpretation():
    print(f"Starting BATCH Smooth Medical XAI Interpreter on: {Config.DEVICE}")

    model = TwoStageAlex3D(drop=0).to(Config.DEVICE) 
    try:
        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        print(f"Successfully loaded full multimodal weights!")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return

    print("Loading cohort dataframe...")
    df = pd.read_pickle(Config.DATA_PKL_PATH)
    
    grad_cam = GradCAM3D(model, model.conv5.conv) 

    print(f"========== Starting batch processing for {len(df)} subjects ==========")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Subjects"):
        subject_id = row['Subject']
        
        clin_features = row['Clinical_Features'][:7]
        clin_tensor = torch.tensor(clin_features, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)

        subject_dir = Config.CLEAN_DATA_DIR / f"sub-{subject_id}"
        if not subject_dir.exists():
            continue
            
        nii_files = list(subject_dir.glob("*_desc-brain.nii.gz"))
        if not nii_files:
            continue
            
        nii_img = nib.load(str(nii_files[0]))
        orig_affine = nii_img.affine 
        orig_data = nii_img.get_fdata().astype(np.float32)
        processed_mri = preprocess_mri(orig_data, Config.TARGET_SHAPE)
        
        base_mri_tensor = torch.from_numpy(processed_mri).unsqueeze(0).unsqueeze(0).to(Config.DEVICE)

        smoothed_cam = np.zeros(Config.TARGET_SHAPE)
        
        for i in range(Config.SMOOTH_SAMPLES):
            noise = torch.randn_like(base_mri_tensor) * Config.NOISE_STD
            noisy_mri = base_mri_tensor + noise
            noisy_mri.requires_grad_(True)
            
            score = model(noisy_mri, clin_tensor)
            
            model.zero_grad()
            score.backward() 
            
            weights = torch.mean(grad_cam.gradients, dim=(2, 3, 4), keepdim=True)
            cam = torch.sum(weights * grad_cam.activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            cam = F.interpolate(cam, size=Config.TARGET_SHAPE, mode='trilinear', align_corners=False)
            cam_np = cam.detach().cpu().numpy().squeeze()
            
            bg_mask = (processed_mri > processed_mri.min() + 1e-3).astype(np.float32)
            cam_np = cam_np * bg_mask
            
            smoothed_cam += cam_np

        smoothed_cam = smoothed_cam / Config.SMOOTH_SAMPLES
        smoothed_cam = (smoothed_cam - smoothed_cam.min()) / (smoothed_cam.max() - smoothed_cam.min() + 1e-8)
        
        heatmap_nii = nib.Nifti1Image(smoothed_cam, orig_affine) 
        save_nii_path = Config.OUTPUT_DIR / f"sub-{subject_id}_SmoothGradCAM.nii.gz"
        nib.save(heatmap_nii, str(save_nii_path))
        
        processed_mri_viz = (processed_mri - processed_mri.min()) / (processed_mri.max() - processed_mri.min())
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        titles = ['Coronal', 'Sagittal', 'Axial']
        coords = Config.SLICE_COORD
        
        for i, ax in enumerate(axes):
            if i == 0:
                mri_slice, cam_slice = processed_mri_viz[coords[0], :, :], smoothed_cam[coords[0], :, :]
            elif i == 1:
                mri_slice, cam_slice = processed_mri_viz[:, coords[1], :], smoothed_cam[:, coords[1], :]
            else:
                mri_slice, cam_slice = processed_mri_viz[:, :, coords[2]], smoothed_cam[:, :, coords[2]]
            
            mri_slice, cam_slice = np.rot90(mri_slice), np.rot90(cam_slice)
            ax.imshow(mri_slice, cmap='gray')
            im = ax.imshow(cam_slice, cmap='jet', alpha=0.5) 
            ax.axis('off')
            ax.set_title(f"{titles[i]} Slice (C:{coords[i]})", fontsize=16)
        
        fig.colorbar(im, ax=axes.ravel().tolist(), label='MCI Smooth Activation')
        save_png_path = Config.OUTPUT_DIR / f"sub-{subject_id}_Smooth_Reasoning.png"
        plt.savefig(save_png_path, bbox_inches='tight', dpi=100) 
        plt.close()

        del base_mri_tensor, clin_tensor, score, noisy_mri, noise, cam, weights
        torch.cuda.empty_cache()
        gc.collect()

    print("All subjects processed successfully!")

if __name__ == "__main__":
    run_interpretation()