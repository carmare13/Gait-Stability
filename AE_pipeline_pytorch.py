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
import zarr
from torch.utils.data import IterableDataset, get_worker_info
import torch.nn.functional as F
from pathlib import Path
import inspect
import time 
import psutil

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
n_timesteps= 100 #cycle is normalized to 100 points 

# ─── Device ────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ─── 1. Dataset ─────────────────────────────────────────────────────────
class GaitBatchIterable(IterableDataset):
    def __init__(self, store_path, batch_size, return_meta=False):
        self._z = zarr.open(store_path, mode="r")["data"]
        self.bs = batch_size
        self.return_meta = return_meta
        self.n = len(self._z) # Total number of individual cycles

    def __len__(self):
        return self.n // self.bs  # number of full batches we will yield   

    def __iter__(self):
        # one per worker
        worker_info = get_worker_info()
        rng = np.random.default_rng(worker_info.id if worker_info else None)

        all_batch_start_idxs = np.arange(0, self.n, self.bs)
        rng.shuffle(all_batch_start_idxs)

        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            num_batches_total = len(all_batch_start_idxs)
            num_batches_per_worker = num_batches_total // num_workers
            remainder_batches = num_batches_total % num_workers

            start_idx = worker_info.id * num_batches_per_worker
            end_idx = (worker_info.id + 1) * num_batches_per_worker
            
            # The last worker handles any remaining batches
            start_idx += min(worker_id, remainder_batches)
            end_idx += min(worker_id, remainder_batches)
            if worker_id < remainder_batches:
                end_idx += 1 # This worker takes one extra batch from the remainder

            worker_batch_start_idxs = all_batch_start_idxs[start_idx:end_idx]
        else:
            # Main process case (num_workers=0) - all batches go to the main process
            worker_batch_start_idxs = all_batch_start_idxs

        # Now, iterate through the batch starting indices assigned to this worker
        for start_of_batch_idx in worker_batch_start_idxs:
            # Construct the actual indices for the current batch
            # This ensures that even the last batch (if smaller) is included
            batch_idxs = np.arange(start_of_batch_idx, min(start_of_batch_idx + self.bs, self.n))

            # No 'pass' needed for 'if len(batch_idxs) < self.bs:' as the slicing already handles it.
            # The 'data' will simply be of the actual size available.

            data = self._z[batch_idxs] # ONE C-read of size len(batch_idxs) × 100 × 326
            
            # Convert NumPy arrays to PyTorch tensors
            feat_np = data[:, :, :321]    # (current_bs,100,321)
            feat    = torch.from_numpy(feat_np).float()
            if self.return_meta:
                meta_np = data[:, :, 321:]  # (current_bs,100,5)
                meta    = torch.from_numpy(meta_np).float()
                yield feat, meta
            else:
                yield feat, feat

# ─── 2. Model Definitions ────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_timesteps, n_vars, latent_dim, dropout=0.2):
        super().__init__()
        self.n_timesteps = n_timesteps

        self.encoder = nn.LSTM(input_size=n_vars, hidden_size=latent_dim,
                               batch_first=True)
        self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim,
                               batch_first=True)
        
        self.hidden_layer = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(latent_dim, n_vars)
        self.apply(self.init_weights)

    def encode(self, x):
        _, (h_n, _) = self.encoder(x)  # h_n: (1, batch, latent_dim)
        z = h_n.squeeze(0)             # (batch, latent_dim)
        return z

    def decode(self, z):
        z_rep = z.unsqueeze(1).repeat(1, self.n_timesteps, 1)  # (batch, seq, latent_dim)
        dec_out, _ = self.decoder(z_rep)                       # (batch, seq, latent_dim)
        h = torch.tanh(self.hidden_layer(dec_out))             # (batch, seq, latent_dim)
        h = self.dropout(h)
        out = self.output_layer(h)                             # (batch, seq, n_vars)
        return out

    def forward(self, x, return_z=False):
        z = self.encode(x)
        out = self.decode(z)
        return (out, z) if return_z else out

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

class BiLSTMAutoencoder(nn.Module):
    def __init__(self, n_timesteps, n_vars, latent_dim, dropout=0.2):
        super().__init__()
        self.n_timesteps = n_timesteps
        
        # Encoder bidireccional
        self.encoder = nn.LSTM(input_size=n_vars, hidden_size=latent_dim,
                               batch_first=True, bidirectional=True)
        # Bottleneck para reducir 2*latent_dim → latent_dim
        self.bottleneck = nn.Linear(2 * latent_dim, latent_dim)
        
        # Decoder bidireccional
        self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim,
                               batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(2 * latent_dim)
        
        # Capa oculta y dropout (trabajan sobre 2*latent_dim)
        self.hidden_layer = nn.Linear(2 * latent_dim, 2 * latent_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Capa de salida
        self.output_layer = nn.Linear(2 * latent_dim, n_vars)

        self.apply(self.init_weights)
    
    def encode(self, x):
        """
        Devuelve el vector latente z para cada muestra de x.
        """
        enc_out, _ = self.encoder(x)             # (batch, seq, 2*latent_dim)
        h_forward = enc_out[:, -1, :latent_dim]
        h_backward = enc_out[:, 0, latent_dim:]
        last = torch.cat([h_forward, h_backward], dim=1)           # (batch, 2*latent_dim)
        z       = self.bottleneck(last)         # (batch, latent_dim)
        return z

    def decode(self, z):
        """
        Reconstruye la secuencia a partir de z, con capa oculta y dropout.
        """
        # (batch, latent_dim) → (batch, seq, latent_dim)
        z_rep = z.unsqueeze(1).repeat(1, self.n_timesteps, 1)
        dec_out, _ = self.decoder(z_rep)         # (batch, seq, 2*latent_dim)
        dec_out = self.norm(dec_out)      # LayerNorm aplicada
        
        # Capa oculta + activación + dropout
        h = torch.tanh(self.hidden_layer(dec_out))   # (batch, seq, 2*latent_dim)
        h = self.dropout(h)
        
        # Capa final de reconstrucción
        out = self.output_layer(h)               # (batch, seq, vars)
        return out

    def forward(self, x, return_z=False):
        z = self.encode(x)
        out = self.decode(z)
        return (out, z) if return_z else out
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
    
    
# ─── 3. Training ────────────────────────────────────────────────────────
def train_autoencoder(model, train_loader, val_loader, run_id, epochs,
                      model_dir='saved_models', log_dir='logs',
                      lr_initial=1e-4, lr_decay_rate=0.98, lr_decay_steps=5000,
                      clip_norm=1.0, patience=5, return_history: bool = False,
                      debug: bool = False,
                      debug_batches: int = 10):
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    model.to(device)

    writer = SummaryWriter(f"{log_dir}/{run_id}")
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=lr_initial, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_decay_rate ** (step // lr_decay_steps)
    )

    best_val = float('inf')
    epochs_no_improve = 0
    process = psutil.Process(os.getpid())
    train_hist, val_hist = [], []

    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        

        if debug:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

        for step, (x, _) in enumerate(train_loader):
            if debug:
                t1 = time.perf_counter()

            x = x.to(device, non_blocking=True)

            if debug:
                torch.cuda.synchronize(device)    # aseguramos que la copia termine
                t2 = time.perf_counter()

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                recon, z = model(x, return_z=True)
                loss = F.mse_loss(recon, x)
                loss += 1e-4 * torch.mean(z ** 2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

            if debug:
                torch.cuda.synchronize(device)
                t3 = time.perf_counter()

                # —— medición de memoria GPU —— 
                used = torch.cuda.memory_allocated(device)           # bytes actuales
                peak = torch.cuda.max_memory_allocated(device)       # bytes pico
                # —— medición de memoria CPU —— 
                mem = process.memory_info().rss                       # RSS en bytes

                print(
                    f"[Epoch {epoch}] Batch {step}: "
                    f"I/O={(t1-t0):.3f}s, memcpy={(t2-t1):.3f}s, comp+back={(t3-t2):.3f}s, "
                    f"GPU_used={used/1e9:.2f}GB, GPU_peak={peak/1e9:.2f}GB, "
                    f"CPU_RSS={mem/1e9:.2f}GB"
                )
                t0 = time.perf_counter()
                if step >= debug_batches-1:
                    break

        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        train_hist.append(train_loss)

        # —— guarda estadísticas de memoria por época —— 
        if debug:
            epoch_gpu_peak = torch.cuda.max_memory_allocated(device)
            epoch_cpu_rss = process.memory_info().rss
            writer.add_scalar('Mem/GPU_peak_GB', epoch_gpu_peak/1e9, epoch)
            writer.add_scalar('Mem/CPU_RSS_GB', epoch_cpu_rss/1e9, epoch)

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    recon_val = model(x_val)
                    loss_val = F.mse_loss(recon_val, x_val)
                val_loss += loss_val.item()
        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        val_hist.append(val_loss)
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # — Early stopping & guardado —
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            best_path = f"{model_dir}/best_ae_{run_id}.pth"
            torch.save(model.state_dict(), f"{model_dir}/best_ae_{run_id}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break    

    final_path = f"{model_dir}/ae_lstm_{run_id}.pth"
    torch.save(model.state_dict(), final_path)
    print(f"[SAVE] Final model → {final_path}")
    writer.close()

    if return_history:
        return train_hist, val_hist



# ─── 4. Evaluación y detección ─────────────────────────────────────────
def evaluate_autoencoder(model, data_loader, device="cpu"):
    """
    Evalúa un autoencoder utilizando MSE, MAE y R².

    Parámetros:
        model : instancia del modelo autoencoder entrenado (PyTorch)
        data_loader : DataLoader con el conjunto de evaluación
        device : "cpu" o "cuda" según corresponda

    Retorna:
        dict con métricas: {"MSE": valor, "MAE": valor, "R2": valor}
    """
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)

            predictions.append(output.cpu())
            targets.append(x_batch.cpu())

    # Convertir a numpy y a forma plana
    y_pred = torch.cat(predictions, dim=0).numpy()
    y_true = torch.cat(targets, dim=0).numpy()

    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    y_true_flat = y_true.reshape(y_true.shape[0], -1)

    mse = F.mse_loss(torch.tensor(y_pred_flat), torch.tensor(y_true_flat)).item()
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    return {"MSE": mse, "MAE": mae, "R2": r2}


#from sklearn.metrics import mean_absolute_error

def evaluate_autoencoder_streaming(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, _ in loader:
            # a CPU device no necesita pin_memory y ya lo tienes desactivado
            x_batch = x_batch.to(device)
            out = model(x_batch)

            # mueve al host y calcula en “batch”
            y_true = x_batch.cpu().numpy().reshape(x_batch.size(0), -1)
            y_pred = out.cpu().numpy().reshape(out.size(0), -1)

            # métricas “sum-of”
            batch_size = y_true.shape[0]
            total_samples += batch_size

            # MSE sumado
            total_mse += ((y_true - y_pred)**2).sum()
            # MAE sumado
            total_mae += abs(y_true - y_pred).sum()

            # liberamos cuanto antes
            del x_batch, out, y_true, y_pred
            torch.cuda.empty_cache()
            import gc; gc.collect()

    mse = total_mse / (total_samples * loader.dataset[0][0].numel())
    mae = total_mae / (total_samples * loader.dataset[0][0].numel())
    return {'mse': mse, 'mae': mae}


# Guardar en archivo
#script_path = Path("/mnt/data/evaluate_autoencoder.py")
#script_path.write_text(inspect.getsource(evaluate_autoencoder))
#script_path.write_text(inspect.getsource(evaluate_autoencoder), encoding='utf-8')

#print(f"Script guardado en: {script_path}")

#str(file_path)

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

