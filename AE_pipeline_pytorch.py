import os
import random
import json
from tqdm import tqdm
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# Asegúrate de que Data_loader.py está en el mismo directorio
from Data_loader import (
    load_subjects_from_json,
      get_all_npy_paths_by_group,
        base_folders
)

# ─── Seeds ─────────────────────────────────────────────────────────────
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
# Garantizar reproducibilidad en cuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ─── Configuración ──────────────────────────────────────────────────────

BATCH_SIZE = 256
NUM_BIOMECHANICAL_VARIABLES = 321
n_timesteps= 100 #cycle is normalized to 100 points 

# ─── Device ────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ─── 1. Dataset ─────────────────────────────────────────────────────────
class LazyGaitDataset(Dataset):
    def __init__(self, npy_paths, n_timesteps=100, n_vars=321):
        self.indexes = []
        self.n_timesteps = n_timesteps
        self.n_vars = n_vars
        self._cache: Dict[str, np.ndarray] = {}
        for p in npy_paths:
            arr = np.load(p, mmap_mode='r')
            for i in range(arr.shape[0]):
                self.indexes.append((p, i))
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self, idx):
        p, i = self.indexes[idx]
        if p not in self._cache:
            self._cache[p] = np.load(p, mmap_mode='r')
        arr = self._cache[p]
        cycle = arr[i, :self.n_timesteps, :self.n_vars].astype(np.float32)
        t = torch.from_numpy(cycle)
        return t, t

def create_dataloader(npy_paths, batch_size=256, is_train=True, **kwargs):
    ds = LazyGaitDataset(npy_paths, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=8,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True,  # workers permanentes
        prefetch_factor=2         # lotes por worker en buffer
    )

class TestGaitDataset(Dataset):
    def __init__(self,
                 npy_paths,
                 subjects_by_group: Dict[str, List[str]],
                 base_folders: Dict[str,str],
                 n_timesteps=100,
                 n_vars=321):
        """
        npy_paths: lista de rutas a .npy (generada con get_all_npy_paths_by_group).
        subjects_by_group: e.g. {"G01": [...ids...], "G03": [...]}.
        base_folders: mapeo de grupo → carpeta raíz donde están los .npy.
        """
        self.indexes = []
        self.n_timesteps = n_timesteps
        self.n_vars = n_vars
        self._cache: Dict[str, np.ndarray] = {}

        for group, subjects in subjects_by_group.items():
            root = base_folders[group]
            for subj in subjects:
                # reconstruye rutas .npy de ese sujeto:
                # (o filtra npy_paths que contengan group y subj)
                subj_paths = [
                    p for p in npy_paths
                    if f"/{group}/" in p and f"/{subj}/" in p
                ]
                for p in subj_paths:
                    arr = np.load(p, mmap_mode='r')
                    for i in range(arr.shape[0]):
                        self.indexes.append({
                            "path": p,
                            "cycle_idx": i,
                            "group": group,
                            "subject": subj
                        })

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        info = self.indexes[idx]
        p = info["path"]
        i = info["cycle_idx"]
        if p not in self._cache:
            self._cache[p] = np.load(p, mmap_mode='r')
        arr = self._cache[p]
        cycle = arr[i, :self.n_timesteps, :self.n_vars].astype(np.float32)
        t = torch.from_numpy(cycle)

        # metadata: 5 campos que puedes serializar como entero o string
        meta = {
            "group":    info["group"],
            "subject":  info["subject"],
            "cycle":    i,
            # añade aquí los otros 2 campos de metadata que necesites
        }
        return t, t, meta
    
def create_test_dataloader(npy_paths,
                           subjects_by_group,
                           base_folders,
                           batch_size=256,
                           **kwargs):
    ds = TestGaitDataset(npy_paths, subjects_by_group, base_folders, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,          # normalmente no barajamos en test
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True,  # workers permanentes
        prefetch_factor=2         # lotes por worker en buffer
    )
    
# ─── 2. Model Definitions ────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_timesteps, n_vars, latent_dim):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.encoder = nn.LSTM(input_size=n_vars, hidden_size=latent_dim,
                               batch_first=True)
        self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim,
                               batch_first=True)
        self.output_layer = nn.Linear(latent_dim, n_vars)
    def forward(self, x):
        # x: (batch, seq, vars)
        _, (h_n, _) = self.encoder(x)
        z = h_n.squeeze(0)  # (batch, latent)
        z_rep = z.unsqueeze(1).repeat(1, self.n_timesteps, 1)
        dec_out, _ = self.decoder(z_rep)
        out = self.output_layer(dec_out)
        return out

class BiLSTMAutoencoder(nn.Module):
    def __init__(self, n_timesteps, n_vars, latent_dim):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.encoder = nn.LSTM(input_size=n_vars, hidden_size=latent_dim,
                               batch_first=True, bidirectional=True)
        self.bottleneck = nn.Linear(2*latent_dim, latent_dim)
        self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim,
                               batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(2*latent_dim, n_vars)
    def encode(self, x):
        """
        Devuelve el vector latente z para cada muestra de x.
        """
        enc_out, _ = self.encoder(x)
        last = enc_out[:, -1, :]       # (batch, 2*latent_dim)
        z    = self.bottleneck(last)   # (batch, latent_dim)
        return z

    def decode(self, z):
        """
        Reconstruye la secuencia a partir de z. 
        Aquí repetimos z en cada timestep como input al decoder.
        """
        # z: (batch, latent_dim) → (batch, 1, latent_dim) → (batch, n_timesteps, latent_dim)
        z_rep = z.unsqueeze(1).repeat(1, self.n_timesteps, 1)
        dec_out, _ = self.decoder(z_rep)
        out = self.output_layer(dec_out)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# ─── 3. Training ────────────────────────────────────────────────────────
def train_autoencoder(model, train_loader, val_loader, run_id, epochs,
                      model_dir='saved_models', log_dir='logs',
                      lr_initial=1e-4, lr_decay_rate=0.98, lr_decay_steps=5000,
                      clip_norm=1.0):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(f"{log_dir}/{run_id}")
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=lr_initial)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_decay_rate ** (step // lr_decay_steps)
    )
    best_val = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.
        for step, (x, _) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                recon = model(x)
                loss = nn.functional.mse_loss(recon, x)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validación
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                with autocast():
                    recon = model(x)
                    loss = nn.functional.mse_loss(recon, x)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{model_dir}/best_ae_{run_id}.pth")

    torch.save(model.state_dict(), f"{model_dir}/ae_lstm_{run_id}.pth")

# ─── 4. Evaluación y detección ─────────────────────────────────────────
def evaluate_and_detect(model, test_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in test_loader:
            # Desempaquetar batch que puede ser (x, _) o solo x
            if isinstance(batch, (list, tuple)) and len(batch) > 1:
                x = batch[0]
            else:
                x = batch
            # Asegurarse de que x está en GPU/CPU según corresponda
            x = x.to(device, non_blocking=True)
            with autocast():
                recon = model(x)
            # reconstrucción vs original
            batch_losses = ((recon - x) ** 2).mean(dim=(1, 2)).cpu().numpy()
            losses.append(batch_losses)
    # Concatenar todos los batches
    losses = np.concatenate(losses, axis=0)
    test_loss = losses.mean()
    print(f"Test reconstruction MSE: {test_loss:.6f}")

    # Cálculo del umbral de anomalías
    p75, p25 = np.percentile(losses, [75, 25])
    threshold = np.median(losses) + 1.5 * (p75 - p25)
    n_anom = (losses > threshold).sum()
    print(f"Detected {n_anom} anomalies out of {len(losses)} (threshold={threshold:.6f})")

    return losses, threshold
# ─── 5. Extracción de latentes ─────────────────────────────────────────
def extract_and_save_latents(
    model,
    loader,
    output_path="latent_features_test.npy",
    device=torch.device("cpu")
):
    """
    Extrae vectores latentes de un encoder LSTM (uni- o bidireccional),
    guardándolos en un .npy y devolviéndolos como ndarray.
    """
    model.to(device).eval()
    latents = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with autocast():
                # Si el modelo tiene un método `encode`, lo usamos
                if hasattr(model, "encode"):
                    # BiLSTMAutoencoder
                    z = model.encode(x)  # (B, latent_dim)

                else:
                    # LSTMAutoencoder unidireccional
                    _, (h_n, _) = model.encoder(x)  # (1, B, H)
                    # Aplanar (batch, hidden)
                    z = h_n.squeeze(0)  # (B, H)
            latents.append(z.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    np.save(output_path, latents)
    print(f"Saved latent features to {output_path}, shape {latents.shape}")
    return latents

# ─── 6. Reconstrucción y métricas ──────────────────────────────────────
def reconstruct_and_evaluate(model_path, data, attr_idx, batch_size, n_timesteps, n_vars, latent_dim):
    model = LSTMAutoencoder(n_timesteps, n_vars, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(data.astype(np.float32)), torch.zeros(len(data))
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=(device.type == 'cuda'),
    )

    recon_list = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with autocast():
                recon = model(x)
            recon_list.append(recon.cpu().numpy())
    recon = np.concatenate(recon_list, axis=0)

    orig = data[:, :, attr_idx]
    recon_sub = recon[:, :, attr_idx]
    mse = ((orig - recon_sub)**2).mean(axis=(0,1))
    mae = np.abs(orig - recon_sub).mean(axis=(0,1))
    rmse = np.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}, recon_sub

