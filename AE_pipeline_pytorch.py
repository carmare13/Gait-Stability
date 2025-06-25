import os
import random
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tfrecord.torch.dataset import TFRecordDataset

import tensorflow as tf

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

BATCH_SIZE = 256
NUM_BIOMECHANICAL_VARIABLES = 321
n_timesteps= 100 #cycle is normalized to 100 points 

# ─── Device ────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── TFRecord ↔ PyTorch DataLoader ────────────────────────────────────────
def _transform(record):
    """Convierte el campo 'data' (bytes) a tensor de forma (n_timesteps, n_vars)."""
    flat = np.frombuffer(record["data"], dtype=np.float32)
    return torch.from_numpy(flat).view(-1, NUM_BIOMECHANICAL_VARIABLES)

def create_tfrecord_dataloader(
    tfrecord_paths: list[str],
    idx_paths:    list[str],
    batch_size:   int = BATCH_SIZE,
    shuffle:      bool = True,
    num_workers:  int = 4,
    pin_memory:   bool = True
) -> DataLoader:
    """
    Devuelve un DataLoader que lee tus shards .tfrecord.gz uno a uno.
    """
    description = {"data": "byte"}
    dataset = TFRecordDataset(
        data_path=tfrecord_paths,
        index_path=idx_paths,
        description=description,
    ).map(_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

# ─── 1. Dataset ─────────────────────────────────────────────────────────
class GaitDataset(Dataset):
    def __init__(self, npy_paths, n_timesteps=100, n_vars=321):
        self.samples = []
        self.n_timesteps = n_timesteps
        self.n_vars = n_vars
        for p in npy_paths:
            arr = np.load(p).astype(np.float32)
            arr = arr[:, :n_timesteps, :n_vars]
            for cycle in arr:
                self.samples.append(cycle)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        cycle = self.samples[idx]
        tensor = torch.from_numpy(cycle)
        return tensor.to(device), tensor.to(device)

def create_dataloader(npy_paths, batch_size=256, is_train=True, **kwargs):
    ds = GaitDataset(npy_paths, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=4, pin_memory=True)

# ─── 2. Model Definitions ────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_timesteps, n_vars, latent_dim, dropout=0.1):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.encoder = nn.LSTM(input_size=n_vars, hidden_size=latent_dim,
                               batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim,
                               batch_first=True, dropout=dropout)
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
    def __init__(self, n_timesteps, n_vars, latent_dim, dropout=0.3):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.encoder = nn.LSTM(input_size=n_vars, hidden_size=latent_dim,
                               batch_first=True, bidirectional=True, dropout=dropout)
        self.bottleneck = nn.Linear(2*latent_dim, latent_dim)
        self.decoder = nn.LSTM(input_size=latent_dim, hidden_size=latent_dim,
                               batch_first=True, bidirectional=True, dropout=dropout)
        self.output_layer = nn.Linear(2*latent_dim, n_vars)
    def forward(self, x):
        enc_out, _ = self.encoder(x)
        last = enc_out[:, -1, :]  # (batch, 2*latent)
        z = self.bottleneck(last) # (batch, latent)
        z_rep = z.unsqueeze(1).repeat(1, self.n_timesteps, 1)
        dec_out, _ = self.decoder(z_rep)
        out = self.output_layer(dec_out)
        return out

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

        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for x, _ in val_loader:
                recon = model(x)
                val_loss += nn.functional.mse_loss(recon, x).item()
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
            x = x.to(device)
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
def extract_and_save_latents(model, test_loader, output_path="latent_features_test.npy"):
    model.eval()
    latents = []
    with torch.no_grad():
        for x, _ in test_loader:
            _, (h_n, _) = model.encoder(x)
            z = h_n.squeeze(0).cpu().numpy()
            latents.append(z)
    latents = np.concatenate(latents)
    np.save(output_path, latents)
    print(f"Saved latent features to {output_path}, shape {latents.shape}")
    return latents

# ─── 6. Reconstrucción y métricas ──────────────────────────────────────
def reconstruct_and_evaluate(model_path, data, attr_idx, batch_size, n_timesteps, n_vars, latent_dim):
    model = LSTMAutoencoder(n_timesteps, n_vars, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ds = torch.utils.data.TensorDataset(torch.from_numpy(data.astype(np.float32)), torch.zeros(len(data)))
    loader = DataLoader(ds, batch_size=batch_size)

    recon_list = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon = model(x).cpu().numpy()
            recon_list.append(recon)
    recon = np.concatenate(recon_list, axis=0)

    orig = data[:, :, attr_idx]
    recon_sub = recon[:, :, attr_idx]
    mse = ((orig - recon_sub)**2).mean(axis=(0,1))
    mae = np.abs(orig - recon_sub).mean(axis=(0,1))
    rmse = np.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}, recon_sub

# ─── 1. TFRecord Conversion ────────────────────────────────────────────

def _bytes_feature(value: bytes) -> tf.train.Feature:
    """
    Auxiliary function to convert bytes to tf.train.Feature 
    """    
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_cycle(cycle: np.ndarray) -> bytes:
    """
    key function to prepare an individual "cycle" (a sequence of data)
    to be saved to a TFRecord
    """
    cycle = cycle.astype(np.float32)
    raw = cycle.tobytes()
    feature = {'data': _bytes_feature(raw)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def write_sharded_tfrecord(npy_paths, output_dir, shard_size):
    """
    Divides the npy list into TFRecords .GZIP,
    each with shard_size cycles
    """
    os.makedirs(output_dir, exist_ok=True)
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    shard_idx = 0
    count = 0
    writer = None

    for p in tqdm(npy_paths, desc="→ Generating shards"):
        arr = np.load(p).astype(np.float32)
        # opcional: recorta a N_TIMESTEPS x NUM_BIOMECHANICAL_VARIABLES
        arr = arr[:, :n_timesteps, :NUM_BIOMECHANICAL_VARIABLES]

        for cycle in arr:
            if writer is None:
                shard_path = os.path.join(output_dir, f"train_shard_{shard_idx:03d}.tfrecord.gz")
                writer = tf.io.TFRecordWriter(shard_path, options=options)
            writer.write(serialize_cycle(cycle))
            count += 1
            if count >= shard_size:
                writer.close()
                shard_idx += 1
                count = 0
                writer = None

    # cierra el último writer si queda abierto
    if writer is not None:
        writer.close()

def convert_npy_to_tfrecord(npy_paths, tfrecord_path):
    """
    Function to create a tfrecord is the data is not that big 
    """
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(tfrecord_path, options) as writer:
        for p in tqdm(npy_paths, desc=f"→ {os.path.basename(tfrecord_path)}"):
            arr = np.load(p).astype(np.float32)
            arr = arr[:, :n_timesteps, :NUM_BIOMECHANICAL_VARIABLES]
            for cycle in arr:
                writer.write(serialize_cycle(cycle))

def write_labeled_tfrecord(
    test_npy,
    output_tfrecord: str,
    label_map: dict,
    n_timesteps: int = 100,
    n_vars: int = 321,
    compression: str = "GZIP"
):
    """
    Escribe un TFRecord con (señal, etiqueta) para validación.

    Args:
      test_npy: dict de la forma {grupo: [rutas a .npy]}
      output_tfrecord: ruta de salida (.tfrecord.gz)
      label_map: mapeo de nombre de grupo → entero de etiqueta
      n_timesteps: número de pasos temporales por ciclo
      n_vars: número de variables biomecánicas (321)
      compression: tipo de compresión ("" o "GZIP")
    """
    options = tf.io.TFRecordOptions(compression_type=compression) \
              if compression else None

    with tf.io.TFRecordWriter(output_tfrecord, options=options) as writer:
        for group, paths in test_npy.items():
            lbl = int(label_map[group])
            for p in paths:
                arr = np.load(p).astype(np.float32)
                # recorta a (n_cycles, n_timesteps, n_vars)
                arr = arr[:, :n_timesteps, :n_vars]

                for cycle in arr:
                    raw = cycle.tobytes()
                    feature = {
                        "data" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lbl]))
                    }
                    ex = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )
                    writer.write(ex.SerializeToString())
    print(f"TFRecord etiquetado creado en: {output_tfrecord}")

# ─── 2. tf.data Pipeline ───────────────────────────────────────────────
def _parse_cycle(example_proto):
    """
    This is the inverse of serialize_cycle.
    It takes a serialized TFRecord and converts it back to a TensorFlow tensor.
    """
    feat_desc = {'data': tf.io.FixedLenFeature([], tf.string)}
    parsed = tf.io.parse_single_example(example_proto, feat_desc)
    flat = tf.io.decode_raw(parsed['data'], tf.float32)
    cycle = tf.reshape(flat, [n_timesteps, NUM_BIOMECHANICAL_VARIABLES])
    return cycle, cycle #la red intenta reconstruir su propia entrada

def parse_for_eval(example_proto):
    feat_desc = {
      'data' : tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feat_desc)
    cycle = tf.io.decode_raw(parsed['data'], tf.float32)
    cycle = tf.reshape(cycle, [n_timesteps, NUM_BIOMECHANICAL_VARIABLES])
    label = parsed['label']
    return cycle, label

def create_tfrecord_dataset(tfrecord_paths, is_training=True):
    """
    builds the complete data pipeline
    """
    dataset = tf.data.TFRecordDataset(
        tfrecord_paths,
        compression_type="GZIP",
        num_parallel_reads=tf.data.AUTOTUNE
    )
    dataset = dataset.map(_parse_cycle, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        #dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=10_000, seed=42, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def make_monolithic_ds(path):
    """
    A simplified version for creating a single TFRecord file dataset
    without the chunking or merging logic
    for validation or test.
    """
    return (
        tf.data.TFRecordDataset(path, compression_type="GZIP")
        .map(_parse_cycle, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )