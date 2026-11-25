import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndimage
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import shap 
import warnings
from tqdm import tqdm # progress bar

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# ================= 1. Rebuild the network architecture (unchanged) =================
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
        self.bn, self.elu, self.drop = nn.BatchNorm3d(out_ch), nn.ELU(inplace=False), nn.Dropout3d(drop)
        self.pool = nn.MaxPool3d(3, 2) if pool else None
    def forward(self, x):
        x = self.drop(self.elu(self.bn(self.conv(x))))
        return self.pool(x) if self.pool else x

class FCBlock(nn.Module):
    def __init__(self, in_f, out_f, drop=0.1):
        super().__init__()
        self.fc, self.bn, self.elu, self.drop = nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.ELU(inplace=False), nn.Dropout(drop)
    def forward(self, x): return self.drop(self.elu(self.bn(self.fc(x))))

class SingleStreamAlex3D(nn.Module):
    def __init__(self, num_clinical=8, drop=0.1): 
        super().__init__()
        self.conv1 = ConvBlock(1, 24, (11,13,11), 4, True, drop)
        self.conv2 = ConvBlock(24, 48, (5,6,5), 1, True, drop)
        self.sep = nn.Sequential(*[nn.Sequential(SeparableConv3d(48,48,(3,3,3),padding=1), nn.BatchNorm3d(48), nn.ELU(), nn.Dropout3d(drop)) for _ in range(3)])
        self.conv5 = ConvBlock(48, 24, (3,4,3), 1, True, drop)
        self.conv6 = ConvBlock(24, 8, (3,4,3), 1, False, drop)
        self.clin_fc1 = FCBlock(num_clinical, 32, drop)
        self.clin_fc2 = FCBlock(32, 10, drop)
        self.img_fc = FCBlock(8, 10, drop) 
        self.fuse_fc = FCBlock(20, 4, drop) 
        self.out = nn.Linear(4, 1)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        
    def forward(self, mri, clin):
        x = self.conv2(self.conv1(mri))
        x = x + self.sep(x)
        img_f = self.pool(self.conv6(self.conv5(x))).view(x.size(0), -1)
        return torch.sigmoid(self.out(self.fuse_fc(torch.cat([self.img_fc(img_f), self.clin_fc2(self.clin_fc1(clin))], dim=1))))

# ================= 2. Prepare device and model =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Starting full SHAP analysis on device: {DEVICE}")

model = SingleStreamAlex3D(num_clinical=8).to(DEVICE)
checkpoint_path = "./Saved_Models/baseline_fold2_epoch10.pth" 
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# ================= 3. Data loading function =================
df_cohort = pd.read_pickle("./Clean_Data/df_cohort_final.pkl")
clinical_feature_names = ['Sex', 'APOE4', 'Age', 'MMSE', 'Global CDR', 'FAQ', 'NPI-Q', 'GDSCALE']

def load_single_mri(subject_id, image_id, data_dir="./Clean_Data"):
    file_path = Path(data_dir) / f"sub-{subject_id}" / f"sub-{subject_id}_{image_id}_desc-brain.nii.gz"
    try:
        img_data = nib.load(str(file_path)).get_fdata().astype(np.float32)
        zoom_factors = [96/img_data.shape[0], 96/img_data.shape[1], 96/img_data.shape[2]]
        resized = ndimage.zoom(img_data, zoom_factors, order=1)
        return (resized - np.mean(resized)) / (np.std(resized) + 1e-8)
    except:
        return np.zeros((96, 96, 96), dtype=np.float32)

# ================= 4. Prepare the background reference set (30 subjects) =================
print("Sampling 30 background subjects...")
bg_df = df_cohort.sample(n=30, random_state=42)
bg_mri = [load_single_mri(r['Subject'], r['Image Data ID']) for _, r in bg_df.iterrows()]
bg_clin = bg_df['Clinical_Features'].tolist()

bg_mri_tensor = torch.tensor(np.array(bg_mri)).unsqueeze(1).to(DEVICE) 
bg_clin_tensor = torch.tensor(np.array(bg_clin), dtype=torch.float32).to(DEVICE) 

explainer = shap.DeepExplainer(model, [bg_mri_tensor, bg_clin_tensor])

# ================= 5. Compute all 990 subjects in batches (avoids OOM) =================
all_clin_shap_values = []
all_clin_features = []
BATCH_SIZE = 10  # 10 subjects per batch keeps memory use stable

print(f"Running batched SHAP analysis on all {len(df_cohort)} subjects...")

for i in tqdm(range(0, len(df_cohort), BATCH_SIZE)):
    batch_df = df_cohort.iloc[i:i+BATCH_SIZE]
    
    batch_mri = [load_single_mri(r['Subject'], r['Image Data ID']) for _, r in batch_df.iterrows()]
    batch_clin = batch_df['Clinical_Features'].tolist()
    
    batch_mri_tensor = torch.tensor(np.array(batch_mri)).unsqueeze(1).to(DEVICE)
    batch_clin_tensor = torch.tensor(np.array(batch_clin), dtype=torch.float32).to(DEVICE)
    
    # compute the current batch
    shap_values = explainer.shap_values([batch_mri_tensor, batch_clin_tensor], check_additivity=False)
    
    # collect clinical SHAP values and the matching raw feature values
    all_clin_shap_values.append(shap_values[1])
    all_clin_features.append(batch_clin_tensor.cpu().numpy())

# merge all batches
final_clin_shap = np.vstack(all_clin_shap_values)
final_clin_feats = np.vstack(all_clin_features)

print("\n========== Plotting clinical feature importance for all 990 subjects ==========")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    final_clin_shap, 
    features=final_clin_feats, 
    feature_names=clinical_feature_names,
    show=False
)
plt.savefig("SHAP_Clinical_Summary_990_ALL.png", bbox_inches='tight', dpi=300)
plt.close()
print("Saved global clinical SHAP plot (990 subjects): SHAP_Clinical_Summary_990_ALL.png")

# save the raw values as CSV for later reference
df_shap_results = pd.DataFrame(final_clin_shap, columns=[f"SHAP_{name}" for name in clinical_feature_names])
df_shap_results.to_csv("SHAP_Clinical_Values_990.csv", index=False)
print("Saved raw SHAP values: SHAP_Clinical_Values_990.csv")