"""
End-to-end LSTM Autoencoder pipeline in a single script,
with logical “module” sections: the data loading (comes from Data_loader.py), TFRecord conversion,
dataset creation, model definition, training, evaluation, anomaly detection,
and latent feature extraction.
"""

import os
import sys
import random
import json
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard, CSVLogger
)

# Import data-loading functions and folder mapping
from Data_loader import (
    load_subjects_from_json,
    get_all_npy_paths_by_group,
    base_folders
)

# Seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

# Optional mixed precision
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")
except ImportError:
    print("Mixed precision not available; using float32")

# ─── 1. TFRecord Conversion “Module” ────────────────────────────────────────────

# These constants must match those in Data_loader.py
NUM_BIOMECHANICAL_VARIABLES = 321
N_TIMESTEPS = 100

def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_cycle(cycle: np.ndarray) -> bytes:
    cycle = cycle.astype(np.float32)
    raw = cycle.tobytes()
    feature = {'data': _bytes_feature(raw)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def write_sharded_tfrecord(npy_paths, output_dir, shard_size=100_000):
    """
    Divide la lista de npy en varios TFRecord comprimidos en GZIP,
    cada uno con hasta shard_size ciclos.
    """
    os.makedirs(output_dir, exist_ok=True)
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    shard_idx = 0
    count = 0
    writer = None

    for p in tqdm(npy_paths, desc="→ Generando shards"):
        arr = np.load(p).astype(np.float32)
        # opcional: recorta a N_TIMESTEPS x NUM_BIOMECHANICAL_VARIABLES
        arr = arr[:, :N_TIMESTEPS, :NUM_BIOMECHANICAL_VARIABLES]

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
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(tfrecord_path, options) as writer:
        for p in tqdm(npy_paths, desc=f"→ {os.path.basename(tfrecord_path)}"):
            arr = np.load(p).astype(np.float32)
            arr = arr[:, :N_TIMESTEPS, :NUM_BIOMECHANICAL_VARIABLES]
            for cycle in arr:
                writer.write(serialize_cycle(cycle))

# ─── 2. tf.data Pipeline “Module” ───────────────────────────────────────────────

BATCH_SIZE = 32

def _parse_cycle(example_proto):
    feat_desc = {'data': tf.io.FixedLenFeature([], tf.string)}
    parsed = tf.io.parse_single_example(example_proto, feat_desc)
    flat = tf.io.decode_raw(parsed['data'], tf.float32)
    cycle = tf.reshape(flat, [N_TIMESTEPS, NUM_BIOMECHANICAL_VARIABLES])
    return cycle, cycle

def create_tfrecord_dataset(tfrecord_paths, is_training=True):
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
    return (
        tf.data.TFRecordDataset(path, compression_type="GZIP")
        .map(_parse_cycle, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
# ─── 3. Model Definition “Module” ───────────────────────────────────────────────
def build_lstm_autoencoder(n_timesteps, n_vars, latent_dim=64, lr=1e-5):
    inputs = Input(shape=(n_timesteps, n_vars))
    x = LSTM(latent_dim, activation='tanh',
            return_sequences=False,
            dropout=0.2, recurrent_dropout=0.2)(inputs)
    x = RepeatVector(n_timesteps)(x)
    x = LSTM(latent_dim, activation='tanh',
            return_sequences=True,
            dropout=0.2, recurrent_dropout=0.2)(x)
    outputs = TimeDistributed(Dense(n_vars, activation='linear'))(x)
    autoencoder = Model(inputs, outputs, name="LSTM_AE")
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    autoencoder.compile(optimizer=opt, loss='mse')
    return autoencoder

# ─── 4. Training “Module” ───────────────────────────────────────────────────────

def train_autoencoder(model, train_ds, val_ds, epochs=100):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_ae_lstm.keras', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        TensorBoard(log_dir='logs/fit'),
        CSVLogger('training_log.csv', append=False)
    ]
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    os.makedirs('saved_models', exist_ok=True)
    model.save('saved_models/ae_lstm.keras') 
    with open('history.json', 'w') as f:
        json.dump(history.history, f)
    return history

# ─── 5. Evaluation & Anomaly Detection “Module” ────────────────────────────────

def evaluate_and_detect(model, test_ds):
    test_loss = model.evaluate(test_ds, verbose=0)
    print(f"Test reconstruction MSE: {test_loss:.6f}")

    losses_ds = test_ds.map(
        lambda x, _: tf.reduce_mean(tf.math.squared_difference(x, model(x)), axis=[1,2]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    all_losses = np.concatenate([b.numpy() for b in losses_ds], axis=0)

    p75, p25 = np.percentile(all_losses, [75, 25])
    threshold = np.median(all_losses) + 1.5 * (p75 - p25)
    n_anom = np.sum(all_losses > threshold)
    print(f"Detected {n_anom} anomalies out of {len(all_losses)} (threshold={threshold:.6f})")
    return all_losses, threshold

# ─── 6. Latent Feature Extraction “Module” ─────────────────────────────────────

def extract_and_save_latents(model, test_ds, output_path="latent_features_test.npy"):
    encoder = Model(inputs=model.input, outputs=model.layers[2].output)
    latent_ds = test_ds.map(
        lambda x, _: encoder(x),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    latents = np.concatenate([l.numpy() for l in latent_ds], axis=0)
    np.save(output_path, latents)
    print(f"Saved latent features to {output_path}, shape {latents.shape}")
    return latents