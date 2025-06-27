import os
import random
import json
from tqdm import tqdm
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast

import time 

""""
# Asegúrate de que Data_loader.py está en el mismo directorio
from Data_loader import (
    load_subjects_from_json,
      get_all_npy_paths_by_group,
        base_folders
)
"""

# ─── Seeds ─────────────────────────────────────────────────────────────
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ─── Configuración de cuDNN ─────────────────────────────────────────────
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
        self.n_timesteps = n_timesteps
        self.n_vars      = n_vars

        self.arrays = {}
        for p in npy_paths:
            arr = np.load(p, mmap_mode='r')
            _ = arr[:]   # fuerza la lectura de *todas* las páginas en cache de SO
            self.arrays[p] = arr


        """
        self.arrays = {
            p: np.load(p, mmap_mode='r')
            for p in npy_paths
        }
        """

        # 2) Prepara la lista de índices
        self.indexes = [
            (p, i)
            for p, arr in self.arrays.items()
            for i in range(arr.shape[0])
        ]



        
        """
        self.indexes = []
        self.n_timesteps = n_timesteps
        self.n_vars = n_vars
        self._cache: Dict[str, np.ndarray] = {}
        for p in npy_paths:
            arr = np.load(p, mmap_mode='r')
            for i in range(arr.shape[0]):
                self.indexes.append((p, i))
"""

    def __len__(self):
        return len(self.indexes)
    

    def __getitem__(self, idx):
        p, i = self.indexes[idx]

        """
        if p not in self._cache:
            self._cache[p] = np.load(p, mmap_mode='r')
        """
        arr = self.arrays[p]  # ya está mapeado
        #arr = self._cache[p]

        cycle = arr[i, :self.n_timesteps, :self.n_vars].astype(np.float32)
        t = torch.from_numpy(cycle)
        return t, t

def create_dataloader(npy_paths, batch_size=256, is_train=True, **kwargs):
    ds = LazyGaitDataset(npy_paths, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=24,
        pin_memory=False,
        #pin_memory=(device.type == 'cuda'),
        persistent_workers=True,  # workers permanentes
        prefetch_factor=4         # lotes por worker en buffer
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
        pin_memory=False,
        #pin_memory=(device.type == 'cuda'),
        persistent_workers=True,  # workers permanentes
        prefetch_factor=2         # lotes por worker en buffer
    )

# class GaitCycleDataset(Dataset):
#     def __init__(self, npy_paths, return_meta=False):
#         """
#         npy_paths: lista de rutas a archivos .npy cada uno con shape (n_cycles, 100, 326)
#         return_meta: si True, __getitem__ retornará (features, metadata)
#         """
#         self.return_meta = return_meta
#         # Cargar memmaps y calcular conteos
#         self._memmaps = []
#         self._counts = []
#         for p in npy_paths:
#             arr = np.load(p, mmap_mode='r')  # no ocupa RAM completa
#             assert arr.ndim == 3 and arr.shape[1] == 100 and arr.shape[2] == 326
#             self._memmaps.append(arr)
#             self._counts.append(arr.shape[0])
#         # índices acumulados para mapear idx -> (file_idx, cycle_idx)
#         self._cum_counts = np.concatenate(([0], np.cumsum(self._counts)))
#         self.total_cycles = int(self._cum_counts[-1])

#     def __len__(self):
#         return self.total_cycles

#     def __getitem__(self, idx):
#         # localizar archivo
#         # cum_counts: [0, c1, c1+c2, ...]
#         file_idx = np.searchsorted(self._cum_counts, idx, side='right') - 1
#         cycle_idx = idx - self._cum_counts[file_idx]
#         data = self._memmaps[file_idx][cycle_idx]  # (100, 326)


#         feat_np = data[:, :321].copy()
#         features = torch.from_numpy(feat_np).float()

#         if self.return_meta:
#             # Metadata: últimas 5 columnas → también copiar
#             meta_np = data[:, 321:].copy()
#             meta = torch.from_numpy(meta_np).float()
#             return features, meta
#         else:
#             return features
class GaitCycleDataset(Dataset):
    def __init__(self, npy_paths, return_meta=False):
        self.return_meta = return_meta
        self._memmaps = []
        self._counts = []
        for p in npy_paths:
            arr = np.load(p, mmap_mode='r')
            assert arr.ndim == 3 and arr.shape[1] == 100 and arr.shape[2] == 326
            self._memmaps.append(arr)
            self._counts.append(arr.shape[0])

        # total number of cycles
        self.total_cycles = sum(self._counts)

        # build lookup tables
        self._file_indices = np.empty(self.total_cycles, dtype=np.int32)
        self._cycle_indices = np.empty(self.total_cycles, dtype=np.int32)
        offset = 0
        for file_idx, cnt in enumerate(self._counts):
            self._file_indices[offset:offset+cnt] = file_idx
            self._cycle_indices[offset:offset+cnt] = np.arange(cnt, dtype=np.int32)
            offset += cnt

    def __len__(self):
        return self.total_cycles

    def __getitem__(self, idx):
        fi = self._file_indices[idx]
        ci = self._cycle_indices[idx]
        data = self._memmaps[fi][ci]   # shape (100, 326)

        # avoid double-copy + dtype conversions
        feat_np = data[:, :321].astype(np.float32, copy=False)
        features = torch.from_numpy(feat_np)
        if self.return_meta:
            meta_np = data[:, 321:].astype(np.float32, copy=False)
            meta = torch.from_numpy(meta_np)
            return features, meta
        else:
            return features

def get_dataloaders(
    train_paths, val_paths, test_paths=None,
    batch_size=4000, num_workers=4, pin_memory=True
):
    """
    Devuelve tus DataLoader para train, val y (opcional) test.
    - train/val: shuffle=True, devuelve tensores [batch,100,321]
    - test: shuffle=False, devuelve (tensores de features, tensores de meta)
    """
    train_ds = GaitCycleDataset(train_paths, return_meta=False)
    val_ds   = GaitCycleDataset(val_paths,   return_meta=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 ,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )

    if test_paths is not None:
        test_ds = GaitCycleDataset(test_paths, return_meta=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader



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
                      clip_norm=1.0, patience=5,
                      debug: bool = False,
                      debug_batches: int = 10):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(f"{log_dir}/{run_id}")
    scaler = GradScaler(device="cuda")
    optimizer = optim.AdamW(model.parameters(), lr=lr_initial)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_decay_rate ** (step // lr_decay_steps)
    )

    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0

        if debug:
            t0 = time.perf_counter()

        for step, (x, _) in enumerate(train_loader):
            if debug:
                t1 = time.perf_counter()

            x = x.to(device, non_blocking=True)
            #x = x.to(device)
            if debug:
                torch.cuda.synchronize()    # aseguramos que la copia termine
                t2 = time.perf_counter()

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                recon = model(x)
                loss = nn.functional.mse_loss(recon, x)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            train_loss += loss.item()

            if debug:
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                print(f"[Epoch {epoch}] Batch {step}: "
                      f"I/O={(t1-t0):.3f}s, memcpy={(t2-t1):.3f}s, comp+back={(t3-t2):.3f}s")
                t0 = t3
                if step >= debug_batches-1:
                    break

        train_loss /= len(train_loader)

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    recon_val = model(x_val)
                    loss_val = nn.functional.mse_loss(recon_val, x_val)
                val_loss += loss_val.item()
        val_loss /= len(val_loader)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # — Early stopping & guardado —
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{model_dir}/best_ae_{run_id}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break    

    torch.save(model.state_dict(), f"{model_dir}/ae_lstm_{run_id}.pth")
    writer.close()

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
            with autocast(device_type=device.type):
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
            with autocast(device_type=device.type):
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
            with autocast(device_type=device.type):
                recon = model(x)
            recon_list.append(recon.cpu().numpy())
    recon = np.concatenate(recon_list, axis=0)

    orig = data[:, :, attr_idx]
    recon_sub = recon[:, :, attr_idx]
    mse = ((orig - recon_sub)**2).mean(axis=(0,1))
    mae = np.abs(orig - recon_sub).mean(axis=(0,1))
    rmse = np.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}, recon_sub

