import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndimage
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

PROJECT_ROOT = Path("/path/to/MCI_Project")
DATA_PKL_PATH = PROJECT_ROOT / "outputs" / "Clean_Data" / "df_cohort_final.pkl"
MRI_DIR = PROJECT_ROOT / "outputs" / "Clean_Data"

MODEL_PATH = PROJECT_ROOT / "outputs" / "Saved_Models" / "Twostage_PRO_fold3_AUC0.9561.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SHAPE = (160, 192, 160)

CLINICAL_NAMES = ['Sex', 'APOE4', 'Age', 'MMSE', 'FAQ', 'NPI-Q', 'GDSCALE']
NUM_SAMPLES = 1000

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

def get_brain_bounding_box(img_data):
    non_zero_indices = np.where(img_data > 0)
    if len(non_zero_indices[0]) == 0: return (0, img_data.shape[0]), (0, img_data.shape[1]), (0, img_data.shape[2])
    pad = 2
    return (max(0, np.min(non_zero_indices[0]) - pad), min(img_data.shape[0], np.max(non_zero_indices[0]) + pad)), \
           (max(0, np.min(non_zero_indices[1]) - pad), min(img_data.shape[1], np.max(non_zero_indices[1]) + pad)), \
           (max(0, np.min(non_zero_indices[2]) - pad), min(img_data.shape[2], np.max(non_zero_indices[2]) + pad))

def preprocess_mri(img_data, target_shape):
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_brain_bounding_box(img_data)
    cropped = img_data[x_min:x_max, y_min:y_max, z_min:z_max]
    scale = min(target_shape[0]/cropped.shape[0], target_shape[1]/cropped.shape[1], target_shape[2]/cropped.shape[2])
    resized = ndimage.zoom(cropped, (scale, scale, scale), order=1)
    pad_x, pad_y, pad_z = target_shape[0] - resized.shape[0], target_shape[1] - resized.shape[1], target_shape[2] - resized.shape[2]
    padded = np.pad(resized, ((pad_x//2, pad_x-pad_x//2), (pad_y//2, pad_y-pad_y//2), (pad_z//2, pad_z-pad_z//2)), mode='constant')
    return (padded - np.mean(padded)) / (np.std(padded) + 1e-8)

def load_patient_data(subject_id, image_id, clin_features):
    file_path = MRI_DIR / f"sub-{subject_id}" / f"sub-{subject_id}_{image_id}_desc-brain.nii.gz"
    if not file_path.exists(): 
        return None, None
    try:
        img_data = nib.load(str(file_path)).get_fdata().astype(np.float32)
        mri_tensor = torch.from_numpy(preprocess_mri(img_data, TARGET_SHAPE)).unsqueeze(0).unsqueeze(0)
        clin_array = np.array(clin_features, dtype=np.float32)[:7]
        clin_tensor = torch.tensor(clin_array).unsqueeze(0)
        return mri_tensor, clin_tensor
    except:
        return None, None

def plot_importance(importance_scores, feature_names):
    sorted_idx = np.argsort(importance_scores)
    sorted_scores = importance_scores[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_names, sorted_scores, color='#4C72B0')
    
    plt.xlabel('Mean Absolute Attribution (Input x Gradient)', fontsize=12)
    plt.title('Clinical Feature Importance for MCI Prediction', fontsize=14, pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (max(sorted_scores)*0.01), bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', va='center', fontsize=10)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    save_path = OUTPUT_DIR / "Clinical_Feature_Importance.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Bar chart successfully saved to: {save_path.name}")

def run_clinical_interpretation():
    print("Starting Clinical Feature Interpreter...")
    
    model = TwoStageAlex3D(num_clinical=7, drop=0.0).to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        print("Successfully loaded model weights.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    df = pd.read_pickle(DATA_PKL_PATH)
    sample_df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42)
    
    all_saliency = []
    all_features = []
    
    print(f"Calculating gradients for {len(sample_df)} sampled patients...")
    for idx, row in sample_df.iterrows():
        mri_t, clin_t = load_patient_data(row['Subject'], row['Image Data ID'], row['Clinical_Features'])
        if mri_t is None: continue
        
        mri_t = mri_t.to(DEVICE)
        clin_t = clin_t.to(DEVICE)
        
        clin_t.requires_grad_(True)
        score = model(mri_t, clin_t)
        
        model.zero_grad()
        score.backward()
        
        saliency = (clin_t.grad * clin_t).detach().cpu().numpy().squeeze()
        all_saliency.append(saliency)
        
        feature_vals = clin_t.detach().cpu().numpy().squeeze()
        all_features.append(feature_vals)

    if not all_saliency:
        print("No valid patient data processed.")
        return

    all_saliency = np.array(all_saliency)
    all_features = np.array(all_features)
    
    global_importance = np.mean(np.abs(all_saliency), axis=0)
    plot_importance(global_importance, CLINICAL_NAMES)
    
    print("Generating SHAP Beeswarm Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        all_saliency, 
        features=all_features, 
        feature_names=CLINICAL_NAMES, 
        show=False
    )
    
    save_path = OUTPUT_DIR / "Clinical_SHAP_Summary.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Classic SHAP plot successfully saved to: {save_path.name}")

if __name__ == "__main__":
    run_clinical_interpretation()