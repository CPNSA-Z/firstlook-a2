import os
import gc
import random
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import nibabel as nib
import scipy.ndimage as ndimage
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# ================= 0. Clear GPU memory and configure real paths =================
torch.cuda.empty_cache()
gc.collect()

PROJECT_ROOT = Path("/path/to/MCI_Project")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FORM_DIR = OUTPUT_DIR / "form"
CLEAN_DATA_DIR = OUTPUT_DIR / "Clean_Data"
MODELS_DIR = OUTPUT_DIR / "Saved_Models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ================= 1. Load data =================
pkl_path = FORM_DIR / "df_cohort_final.pkl"
try:
    df_cohort_final = pd.read_pickle(pkl_path)
    print(f"Successfully loaded clinical data with {len(df_cohort_final)} subjects.")
except FileNotFoundError:
    print(f"Error: File {pkl_path} not found. Please ensure the data has been generated.")
    raise

# ================= 2. Preprocessing and enhanced data augmentation =================
def get_brain_bounding_box(img_data):
    non_zero_indices = np.where(img_data > 0)
    if len(non_zero_indices[0]) == 0:
        return (0, img_data.shape[0]), (0, img_data.shape[1]), (0, img_data.shape[2])
    x_min, x_max = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    y_min, y_max = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    z_min, z_max = np.min(non_zero_indices[2]), np.max(non_zero_indices[2])
    pad = 2
    return (max(0, x_min - pad), min(img_data.shape[0], x_max + pad)), \
           (max(0, y_min - pad), min(img_data.shape[1], y_max + pad)), \
           (max(0, z_min - pad), min(img_data.shape[2], z_max + pad))

def preprocess_mri(img_data, target_shape=(160, 192, 160)):
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_brain_bounding_box(img_data)
    cropped_data = img_data[x_min:x_max, y_min:y_max, z_min:z_max]
    scale = min(target_shape[0] / cropped_data.shape[0],
                target_shape[1] / cropped_data.shape[1],
                target_shape[2] / cropped_data.shape[2])
    resized_data = ndimage.zoom(cropped_data, (scale, scale, scale), order=1)
    
    pad_x = target_shape[0] - resized_data.shape[0]
    pad_y = target_shape[1] - resized_data.shape[1]
    pad_z = target_shape[2] - resized_data.shape[2]
    padded_data = np.pad(resized_data,
                         ((pad_x//2, pad_x-pad_x//2),
                          (pad_y//2, pad_y-pad_y//2),
                          (pad_z//2, pad_z-pad_z//2)),
                         mode='constant')
    return (padded_data - np.mean(padded_data)) / (np.std(padded_data) + 1e-8)

class Compose3D:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms: img = t(img)
        return img

class RandomFlip3D:
    def __init__(self, prob=0.5, axis=0): self.prob, self.axis = prob, axis
    def __call__(self, img): return np.flip(img, axis=self.axis).copy() if random.random() < self.prob else img

class RandomRotate3D:
    def __init__(self, prob=0.5, angle_range=(-10, 10)): self.prob, self.angle_range = prob, angle_range
    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(self.angle_range[0], self.angle_range[1])
            img = ndimage.rotate(img, angle, axes=(0,1), reshape=False, order=1, mode='nearest')
        return img

# Gaussian noise
class GaussianNoise3D:
    def __init__(self, prob=0.3, mean=0., std=0.05): self.prob, self.mean, self.std = prob, mean, std
    def __call__(self, img):
        if random.random() < self.prob:
            img = img + np.random.normal(self.mean, self.std, img.shape).astype(np.float32)
        return img

# Elastic deformation
class ElasticDeformation3D:
    def __init__(self, prob=0.3, alpha=15, sigma=3): self.prob, self.alpha, self.sigma = prob, alpha, sigma
    def __call__(self, img):
        if random.random() < self.prob:
            shape = img.shape
            dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dz = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))
            img = ndimage.map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)
        return img

class ADNI_MCI_Dataset(Dataset):
    def __init__(self, df_cohort, data_dir, target_shape=(160, 192, 160), is_train=True):
        self.df = df_cohort.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.transform = Compose3D([
            RandomFlip3D(prob=0.5, axis=0),
            RandomRotate3D(prob=0.5),
            GaussianNoise3D(prob=0.3),
            ElasticDeformation3D(prob=0.3)
        ]) if is_train else None

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = self.data_dir / f"sub-{row['Subject']}" / f"sub-{row['Subject']}_{row['Image Data ID']}_desc-brain.nii.gz"
        try:
            mri_data = nib.load(str(file_path)).get_fdata().astype(np.float32)
            mri_processed = preprocess_mri(mri_data, self.target_shape)
        except Exception:
            mri_processed = np.zeros(self.target_shape, dtype=np.float32)
        if self.transform: mri_processed = self.transform(mri_processed)
        return torch.from_numpy(mri_processed).unsqueeze(0), \
               torch.tensor(row['Clinical_Features'], dtype=torch.float32), \
               torch.tensor(row['Label'], dtype=torch.float32)

# ================= 3. CBAM + FiLM architecture =================
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

class FiLMFusion(nn.Module):
    def __init__(self, img_dim, clin_dim):
        super().__init__()
        self.film_gen = nn.Linear(clin_dim, img_dim * 2)
    def forward(self, img_f, clin_f):
        film_params = self.film_gen(clin_f)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        return img_f * gamma + beta

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
        self.fc, self.bn, self.elu, self.drop = nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.ELU(inplace=True), nn.Dropout(drop)
    def forward(self, x): return self.drop(self.elu(self.bn(self.fc(x))))

class TwoStageAlex3D(nn.Module):
    def __init__(self, num_clinical=7, drop=0.1): 
        super().__init__()
        self.conv1 = ConvBlock(1, 24, (11,13,11), 4, True, drop)
        self.conv2 = ConvBlock(24, 48, (5,6,5), 1, True, drop)
        self.cbam = CBAM_3D(channels=48)
        self.sep = nn.Sequential(*[
            nn.Sequential(SeparableConv3d(48,48,(3,3,3),padding=1),
                          nn.BatchNorm3d(48), nn.ELU(), nn.Dropout3d(drop))
            for _ in range(3)
        ])
        self.conv5 = ConvBlock(48, 24, (3,4,3), 1, True, drop)
        self.conv6 = ConvBlock(24, 8, (3,4,3), 1, False, drop)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        
        self.img_fc = FCBlock(8, 10, drop)
        self.img_only_out = nn.Linear(10, 1)

        self.clin_fc1 = FCBlock(num_clinical, 32, drop)
        self.clin_fc2 = FCBlock(32, 10, drop)
        
        self.film_fusion = FiLMFusion(img_dim=10, clin_dim=10)
        self.out = nn.Linear(10, 1)

    def forward_image_only(self, mri):
        x = self.conv2(self.conv1(mri))
        x = self.cbam(x)
        x = x + self.sep(x)
        img_f = self.pool(self.conv6(self.conv5(x))).view(x.size(0), -1)
        img_embed = self.img_fc(img_f)
        return torch.sigmoid(self.img_only_out(img_embed))

    def forward(self, mri, clin):
        x = self.conv2(self.conv1(mri))
        x = self.cbam(x)
        x = x + self.sep(x)
        img_f = self.pool(self.conv6(self.conv5(x))).view(x.size(0), -1)
        img_embed = self.img_fc(img_f)
        
        clin_embed = self.clin_fc2(self.clin_fc1(clin))
        fused = self.film_fusion(img_embed, clin_embed)
        return torch.sigmoid(self.out(fused))

# ================= 4. Focal Loss training configuration =================
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        bce_loss = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        if self.pos_weight is not None:
            weight_mask = targets * self.pos_weight + (1 - targets)
            focal_loss = focal_loss * weight_mask
            
        return focal_loss.mean()

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 4
    EPOCHS_STAGE1 = 30
    EPOCHS_STAGE2 = 20

def run_two_stage_pipeline(df_cohort, clean_dir, models_dir):
    print(f"Starting full enhanced training pipeline... Device: {Config.DEVICE}")
    print(f"Configuration: Stage 1 = {Config.EPOCHS_STAGE1} Epochs | Stage 2 = {Config.EPOCHS_STAGE2} Epochs")
    
    train_dataset = ADNI_MCI_Dataset(df_cohort, data_dir=clean_dir, target_shape=(160, 192, 160), is_train=True)
    val_dataset   = ADNI_MCI_Dataset(df_cohort, data_dir=clean_dir, target_shape=(160, 192, 160), is_train=False)
    
    labels = df_cohort['Label'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pos_weight = torch.tensor((labels == 0).sum() / (labels == 1).sum()).to(Config.DEVICE)
    
    criterion = BinaryFocalLoss(gamma=2.0, pos_weight=pos_weight)
    
    # ALL LOGIC MUST BE INSIDE THIS FOLD LOOP
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print("\n=============================================")
        print(f"========== Starting training Fold {fold + 1} ==========")
        print("=============================================")
        
        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=Config.BATCH_SIZE,
                                  shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=Config.BATCH_SIZE,
                                shuffle=False, num_workers=4, pin_memory=True)
        
        model = TwoStageAlex3D(num_clinical=7).to(Config.DEVICE)
        
        # Stage 1: image-only
        print("\n[Stage 1] Image-only training begins (with CBAM + elastic deformation) ...")
        img_params = list(model.conv1.parameters()) + list(model.conv2.parameters()) + \
                     list(model.cbam.parameters()) + \
                     list(model.sep.parameters()) + list(model.conv5.parameters()) + \
                     list(model.conv6.parameters()) + list(model.img_fc.parameters()) + \
                     list(model.img_only_out.parameters())
        optimizer_s1 = Adam(img_params, lr=Config.LEARNING_RATE, weight_decay=1e-4)
        
        for epoch in range(Config.EPOCHS_STAGE1):
            model.train()
            t_loss, corr, tot = 0, 0, 0
            optimizer_s1.zero_grad()
            
            for i, (mri, clin, lbl) in enumerate(tqdm(train_loader, desc=f"S1 Epoch {epoch+1}", leave=False)):
                mri, lbl = mri.to(Config.DEVICE), lbl.to(Config.DEVICE).unsqueeze(1)
                
                out = model.forward_image_only(mri)
                loss = criterion(out, lbl) / Config.ACCUMULATION_STEPS
                loss.backward()
                
                if (i + 1) % Config.ACCUMULATION_STEPS == 0:
                    optimizer_s1.step()
                    optimizer_s1.zero_grad()
                    
                t_loss += (loss.item() * Config.ACCUMULATION_STEPS) * mri.size(0)
                corr += ((out > 0.5) == lbl).sum().item()
                tot += lbl.size(0)
            print(f"Stage 1 - Epoch {epoch+1:02d} | Image Focal Loss: {t_loss/tot:.4f} - Acc: {corr/tot:.4f}")

        # Stage 2: multimodal fusion (FiLM)
        print("\n[Stage 2] Multimodal modulation fusion begins (FiLM) ...")
        for param in img_params:
            param.requires_grad = False

        fuse_params = list(model.clin_fc1.parameters()) + list(model.clin_fc2.parameters()) + \
                      list(model.film_fusion.parameters()) + list(model.out.parameters())
        optimizer_s2 = Adam(fuse_params, lr=Config.LEARNING_RATE)

        best_auc = 0.0
        best_model_state = None

        for epoch in range(Config.EPOCHS_STAGE2):
            model.train()
            model.conv1.eval(); model.conv2.eval(); model.cbam.eval(); model.sep.eval(); model.conv5.eval(); model.conv6.eval()
        
            t_loss, corr, tot = 0, 0, 0
            optimizer_s2.zero_grad()
        
            for i, (mri, clin, lbl) in enumerate(tqdm(train_loader, desc=f"S2 Epoch {epoch+1}", leave=False)):
                mri, clin, lbl = mri.to(Config.DEVICE), clin.to(Config.DEVICE), lbl.to(Config.DEVICE).unsqueeze(1)
            
                out = model(mri, clin)
                loss = criterion(out, lbl) / Config.ACCUMULATION_STEPS
                loss.backward()
            
                if (i + 1) % Config.ACCUMULATION_STEPS == 0:
                    optimizer_s2.step()
                    optimizer_s2.zero_grad()
                
                t_loss += (loss.item() * Config.ACCUMULATION_STEPS) * mri.size(0)
                corr += ((out > 0.5) == lbl).sum().item()
                tot += lbl.size(0)
        
            model.eval()
            v_loss, preds, lbls = 0, [], []
            with torch.no_grad():
                for mri, clin, lbl in val_loader:
                    mri, clin, lbl = mri.to(Config.DEVICE), clin.to(Config.DEVICE), lbl.to(Config.DEVICE).unsqueeze(1)
                    out = model(mri, clin)
                    v_loss += criterion(out, lbl).item() * mri.size(0)
                    preds.extend(out.cpu().numpy().flatten())
                    lbls.extend(lbl.cpu().numpy().flatten())
        
            preds, lbls = np.array(preds), np.array(lbls)
            v_loss /= len(lbls)
            v_acc = accuracy_score(lbls, (preds > 0.5).astype(int))
            try:
                v_auc = auc(*roc_curve(lbls, preds)[:2])
            except:
                v_auc = 0.5
        
            print(f"Stage 2 - Epoch {epoch+1:02d} | Train Focal Loss: {t_loss/tot:.4f} | Val Focal Loss: {v_loss:.4f} - Acc: {v_acc:.4f} - AUC: {v_auc:.4f}")
        
            # Only store model weights in memory when AUC improves
            if v_auc > best_auc:
                best_auc = v_auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # After finishing the loop for ONE fold, write the best model state to disk
        if best_model_state is not None:
            save_path = MODELS_DIR / f"Twostage_PRO_fold{fold+1}_AUC{best_auc:.4f}.pth"
            torch.save({'model_state_dict': best_model_state, 'best_auc': best_auc}, save_path)
            print(f"Fold {fold + 1} completed. Best AUC: {best_auc:.4f}, saved to: {save_path.name}")

if __name__ == "__main__":
    print("Preparing to start (full enhanced version)...")
    try:
        run_two_stage_pipeline(df_cohort_final, clean_dir=CLEAN_DATA_DIR, models_dir=MODELS_DIR)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\nOOM detected. The model is too heavy. Please set BATCH_SIZE to 1.")
        else:
            print(f"\nRuntime error occurred: {e}") 
if __name__ == "__main__":
    print("Starting training (full run)...")
    try:
        run_two_stage_pipeline(df_cohort_final, clean_dir=CLEAN_DATA_DIR, models_dir=MODELS_DIR)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n🚨 OOM! The model is too heavy. Please set BATCH_SIZE to 1.")
        else:
            print(f"\nRuntime error: {e}")