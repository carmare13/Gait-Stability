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
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
        self.n = len(self._z)

    def __len__(self):
        return self.n // self.bs

    def __iter__(self):
        worker_info = get_worker_info()
        rng = np.random.default_rng(worker_info.id if worker_info else None)

        all_batch_start_idxs = np.arange(0, self.n, self.bs)
        rng.shuffle(all_batch_start_idxs)

        if worker_info:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            total_batches = len(all_batch_start_idxs)
            per_worker = total_batches // num_workers
            rem = total_batches % num_workers

            start = worker_id * per_worker + min(worker_id, rem)
            end   = start + per_worker + (1 if worker_id < rem else 0)
            batch_starts = all_batch_start_idxs[start:end]
        else:
            batch_starts = all_batch_start_idxs

        for s in batch_starts:
            batch_idxs = np.arange(s, min(s + self.bs, self.n))
            data = self._z[batch_idxs]              # (bs, 100, 326)
            feat_np = data[:, :, :321]              # (bs, 100, 321)
            feat    = torch.from_numpy(feat_np).float()
            # —— aquí permutamos para (bs, 321, 100):
            #feat = feat.permute(0, 2, 1)

            if self.return_meta:
                meta_np = data[:, :, 321:]          # (bs, 100, 5)
                meta    = torch.from_numpy(meta_np).float()
                # opcional: permutar meta también si quieres (bs, 5, 100)
                #meta = meta.permute(0, 2, 1)
                yield feat, meta
            else:
                yield feat, feat

# ─── 2. Model Definitions ────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_timesteps, n_vars, latent_dim, dropout=0.4):
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
    def __init__(self, n_timesteps, n_vars, latent_dim, dropout=0.4):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.latent_dim = latent_dim

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
        h_forward = enc_out[:, -1, :self.latent_dim]
        h_backward = enc_out[:, 0, self.latent_dim:]
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

class GaitAutoencoder(nn.Module):
    def __init__(self, input_channels, seq_length, latent_dim):
        super().__init__()
        self.seq_length = seq_length

        # 1) convolucional
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_channels, 16, 3, stride=2, padding=1),  # 100→50
            nn.ReLU(True),
            nn.Conv1d(16, 32, 3, stride=2, padding=1),              # 50→25
            nn.ReLU(True),
        )

        # calculamos flat_dim dinámicamente
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, seq_length)
            c, l = self.encoder_conv(dummy).shape[1:]
            flat_dim = c * l

        # 2) encoder/decoder FC
        self.encoder_fc = nn.Linear(flat_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, flat_dim)

        # 3) decoder conv vía unflatten
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (c, l)),                                 # (batch,32,25)
            nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1, output_padding=1),  # →50
            nn.ReLU(True),
            nn.ConvTranspose1d(16, input_channels, 3, stride=2, padding=1, output_padding=1),  # →100
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, 321, 100)
        x_enc = self.encoder_conv(x)
        flat  = x_enc.view(x_enc.size(0), -1)
        z     = self.encoder_fc(flat)            # (batch, latent_dim)

        up    = self.decoder_fc(z)               # (batch, flat_dim)
        x_recon = self.decoder_conv(up)          # (batch, 321, 100)
        # ya sale con 100 pasos, no hace falta recortar
        return x_recon, z


class LSTMConvAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, n_timesteps, dropout=0.4):
        super().__init__()
        self.n_timesteps = n_timesteps

        # Encoder: capas convolucionales
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)

        # LSTM para capturar dependencias temporales
        self.lstm = nn.LSTM(128, latent_dim, batch_first=True)

        # Decoder: LSTM + convoluciones
        self.decoder_lstm = nn.LSTM(latent_dim, 128, batch_first=True)

        # No necesitamos una capa lineal para ajustar la salida de LSTM
        # El tamaño de la salida del LSTM (latent_dim) ya es compatible con el decoder LSTM

        self.deconv1 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.Conv1d(64, input_dim, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(dropout)

    def encode(self, x):
        x = x.permute(0, 2, 1)  # Cambiar el formato a (batch, n_channels, n_timesteps)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Volver a la forma original (batch, seq_len, features)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)

    def decode(self, z):
        # Repetir el vector latente z para cada timestep
        z_rep = z.unsqueeze(1).repeat(1, self.n_timesteps, 1)

        # Salida del decoder LSTM
        dec_out, _ = self.decoder_lstm(z_rep)

        # --- INICIO DE LA CORRECCIÓN ---

        # Permutar de (batch, seq_len, features) a (batch, features, seq_len)
        # para que sea compatible con Conv1d
        dec_out = dec_out.permute(0, 2, 1)

        # Aplicar las deconvoluciones
        dec_out = F.relu(self.deconv1(dec_out))
        dec_out = self.deconv2(dec_out)

        # Permutar de vuelta al formato original (batch, seq_len, features)
        # para que coincida con la forma del tensor de entrada 'x'
        dec_out = dec_out.permute(0, 2, 1)

        # --- FIN DE LA CORRECCIÓN ---

        return dec_out

    # def decode(self, z):
    #     # Ahora la salida del LSTM tiene el tamaño esperado de latent_dim
    #     z_rep = z.unsqueeze(1).repeat(1, self.n_timesteps, 1)  # Repetir z para cada timestep
    #     dec_out, _ = self.decoder_lstm(z_rep)  # El tamaño de z es latent_dim, que es 256
    #     dec_out = F.relu(self.deconv1(dec_out))
    #     dec_out = self.deconv2(dec_out)
    #     return dec_out

    def forward(self, x, return_z=False):
        z = self.encode(x)
        recon = self.decode(z)
        return (recon, z) if return_z else recon







# ─── 3. Training ────────────────────────────────────────────────────────
# def contrastive_loss(x1, x2, label, margin=1.0):
#     # Distancia euclidiana entre las representaciones latentes
#     euclidean_distance = F.pairwise_distance(x1, x2, keepdim=True)
#     loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
#                                   (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
#     return loss_contrastive


def train_autoencoder(model, train_loader, val_loader, run_id, epochs,
                      model_dir='saved_models', log_dir='logs',
                      lr_initial=1e-3, lr_decay_rate=0.98, lr_decay_steps=5000,
                      clip_norm=1.0, patience=5, return_history: bool = False,
                      debug: bool = False,
                      debug_batches: int = 10):

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    model.to(device)

    writer = SummaryWriter(f"{log_dir}/{run_id}")
    scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=lr_initial, weight_decay=5e-4)
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

             # Dentro del ciclo de entrenamiento
            # optimizer.zero_grad()
            # with autocast(device_type=device.type):
            #     recon, z = model(x, return_z=True)

            #     # Pérdida de reconstrucción (mse) + pérdida contrastiva
            #     loss_mse = F.mse_loss(recon, x)
            #     loss_contrastive = contrastive_loss(z, z_anchor, label)  # Calcula la pérdida contrastiva entre pares de ciclos

            #     # Total loss = mse + contrastive loss
            #     loss = loss_mse + loss_contrastive
            #     loss.backward()
            #     optimizer.step()



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
    """
    Evalúa un autoencoder en batches, calculando MSE y MAE de forma acumulativa.

    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    per_sample_elems = None

    with torch.no_grad():
        for batch in loader:
            # Extraer x_batch de (x_batch, _) o directamente x_batch
            x_batch = batch[0] if isinstance(batch, (list,tuple)) else batch
            x_batch = x_batch.to(device, non_blocking=True)

            out = model(x_batch)
            # Si tu forward devuelve tupla, extrae la reconstrucción
            recon = out[0] if isinstance(out, tuple) else out

            # Llevar a NumPy y aplanar
            y_true = x_batch.cpu().numpy().reshape(x_batch.size(0), -1)
            y_pred = recon.cpu().numpy().reshape(recon.size(0), -1)

            # Inicializar per_sample_elems la primera vez
            if per_sample_elems is None:
                per_sample_elems = y_true.shape[1]

            # Acumular
            batch_size = y_true.shape[0]
            total_samples += batch_size
            total_mse     += ((y_true - y_pred)**2).sum()
            total_mae     += np.abs(y_true - y_pred).sum()

    # Ahora sí podemos calcular promedios correctamente
    num_elements = total_samples * per_sample_elems
    mse = total_mse / num_elements
    mae = total_mae / num_elements

    return {'mse': mse, 'mae': mae}


def evaluate_and_detect(model, test_loader):
    """
    Evalúa un autoencoder y detecta anomalías basándose en la reconstrucción.

    """
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

