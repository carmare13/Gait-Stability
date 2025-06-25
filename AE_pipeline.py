"""
End-to-end LSTM Autoencoder pipeline
with logical “module” sections: the data loading (comes from Data_loader.py),
TFRecord conversion,
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
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import regularizers, initializers 
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
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

# Mixed precision: to acelerate training & reduce memory 
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled")
except ImportError:
    print("Mixed precision not available; using float32")

# ─── 1. TFRecord Conversion ────────────────────────────────────────────
NUM_BIOMECHANICAL_VARIABLES = 321
n_timesteps= 100 #cycle is normalized to 100 points 

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
BATCH_SIZE = 256
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

# ─── 3. Model Definition ───────────────────────────────────────────────
def r2(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ss_total = K.sum(K.square(y_true - K.mean(y_true)))
    ss_residual = K.sum(K.square(y_true - y_pred))
    return 1 - (ss_residual / ss_total)

def build_lstm_autoencoder(
    n_timesteps,
    n_vars,
    latent_dim,
    enc_activation='tanh',
    dec_activation='tanh',
    dense_activation='linear',
    recurrent_activation='sigmoid',
    dropout=0.1,
    recurrent_dropout=0.1,
    l2_reg=0.001
):
    inputs = Input(shape=(n_timesteps, n_vars))
    x = LSTM(
        latent_dim,
        activation=enc_activation,
        recurrent_activation=recurrent_activation,
        name="encoder_lstm", 
        return_sequences=False,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_initializer=initializers.GlorotNormal(),  
        recurrent_initializer=initializers.GlorotNormal(),
        kernel_regularizer=regularizers.l2(l2_reg)
    )(inputs)

    x = RepeatVector(n_timesteps)(x)
    x = LSTM(
        latent_dim,
        activation=dec_activation,
        recurrent_activation=recurrent_activation,
        name="decoder_lstm",  
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        kernel_initializer=initializers.GlorotNormal(),
        recurrent_initializer=initializers.GlorotNormal(),
        kernel_regularizer=regularizers.l2(l2_reg) 
    )(x)
    outputs = TimeDistributed(Dense(n_vars, activation=dense_activation,
                                    kernel_regularizer=regularizers.l2(l2_reg)))(x)
    
    autoencoder = Model(inputs, outputs, name="LSTM_AE")
    return autoencoder

def build_bilstm_autoencoder(
    n_timesteps,
    n_vars,
    latent_dim,
    enc_activation='tanh',
    dec_activation='tanh',
    dense_activation='linear',
    recurrent_activation='sigmoid',
    dropout=0.3,
    recurrent_dropout=0.3,
    l2_reg=0.001
):
    """
    Autoencoder with bidirectional encoder and decoder.
    The encoder is a BiLSTM that concatenates both directions
    (2 × latent\_dim), then optionally we reduce it to latent\_dim with a Dense layer.

    """

    inputs = Input(shape=(n_timesteps, n_vars), name="ae_input")

    # — Encoder Bi-LSTM —
    # Salida será (batch, 2*latent_dim)
    x = Bidirectional(
        LSTM(
            latent_dim,
            activation=enc_activation,
            recurrent_activation=recurrent_activation,
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_initializer=initializers.GlorotNormal(),
            recurrent_initializer=initializers.GlorotNormal(),
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        name="encoder_bilstm"
    )(inputs)

    # Reducimos de 2*latent_dim a latent_dim para el bottleneck puro
    x = Dense(
        latent_dim,
        activation='linear',
        name="bottleneck_dense",
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)

    # — Decoder —
    # Volvemos a Expandir al número de timesteps
    x = RepeatVector(n_timesteps, name="bottleneck_repeat")(x)

    # Bi-LSTM en decoder (return_sequences=True)
    x = Bidirectional(
        LSTM(
            latent_dim,
            activation=dec_activation,
            recurrent_activation=recurrent_activation,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_initializer=initializers.GlorotNormal(),
            recurrent_initializer=initializers.GlorotNormal(),
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        name="decoder_bilstm"
    )(x)

    # Capa final que recupera las n_vars originales
    outputs = TimeDistributed(
        Dense(n_vars, activation=dense_activation,
              kernel_regularizer=regularizers.l2(l2_reg)),
        name="decoder_output"
    )(x)

    return Model(inputs, outputs, name="BiLSTM_AE")

# ─── 4. Training ───────────────────────────────────────────────────────

def train_autoencoder(
    model,
    train_ds,
    val_ds,
    run_id: str,               
    epochs,
    steps_per_epoch,       # <-- nuevo param
    validation_steps,      # <-- nuevo param
    model_dir='saved_models',
    log_dir_root='logs/fit',
    csv_log_root='training_log',
    lr_initial=1e-4,
    lr_decay_rate=0.98,
    lr_decay_steps=5000,   
    clipnorm=1.0
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir_root, exist_ok=True)

    chkpt_path   = f"{model_dir}/best_ae_{run_id}.keras"
    tb_log_dir   = f"{log_dir_root}_{run_id}"
    csv_log_path = f"{csv_log_root}_{run_id}.csv"
    final_model  = f"{model_dir}/ae_lstm_{run_id}.keras"


    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(chkpt_path, save_best_only=True, monitor='val_loss'),
        #ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        TensorBoard(log_dir=tb_log_dir),
        CSVLogger(csv_log_path, append=False)
    ]

    lr_schedule = ExponentialDecay(
        initial_learning_rate=lr_initial,
        decay_steps=lr_decay_steps,
        decay_rate=lr_decay_rate,
        staircase=True
    )
    
    base_opt = tf.keras.optimizers.AdamW(learning_rate=lr_schedule,
                                            clipnorm=clipnorm)
    opt = LossScaleOptimizer(base_opt)

    model.compile(
        optimizer=opt,
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(), r2]
    )
    

    history = model.fit(
        train_ds,
        epochs=epochs,
        #steps_per_epoch=steps_per_epoch, 
        validation_data=val_ds,
        #validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(final_model) 
    with open(f'history_{run_id}.json', 'w') as f:
        json.dump(history.history, f)
    return history

# ─── 5. Evaluation & Anomaly Detection ────────────────────────────────
def evaluate_and_detect(model, test_ds):
    results = model.evaluate(test_ds, verbose=0)
    test_loss = results[0]  # first element is the loss
    print(f"Test reconstruction MSE: {test_loss:.6f}")

    losses_ds = test_ds.map(
        lambda x, _: tf.reduce_mean(
            tf.math.squared_difference(
                tf.cast(x, tf.float32),  
                tf.cast(model(x), tf.float32) 
            ), axis=[1, 2]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    all_losses = np.concatenate([b.numpy() for b in losses_ds], axis=0)

    p75, p25 = np.percentile(all_losses, [75, 25])
    threshold = np.median(all_losses) + 1.5 * (p75 - p25)
    n_anom = np.sum(all_losses > threshold)
    print(f"Detected {n_anom} anomalies out of {len(all_losses)} (threshold={threshold:.6f})")
    return all_losses, threshold

# ─── 6. Latent Feature Extraction ─────────────────────────────────────

def extract_and_save_latents(model, test_ds, output_path="latent_features_test.npy"):
    encoder = Model(inputs=model.input, outputs=model.layers[1].output)
    latent_ds = test_ds.map(
        lambda x, _: encoder(x),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    latents = np.concatenate([l.numpy() for l in latent_ds], axis=0)
    # verify latent shape is (batch_size, latent_dim)
    latent_dim = encoder.output_shape[-1]
    assert latents.ndim == 2 and latents.shape[1] == latent_dim, (
        f"Latents have unexpected shape {latents.shape}, expected (_, {latent_dim})"
    )
    np.save(output_path, latents)
    print(f"Saved latent features to {output_path}, shape {latents.shape}")
    return latents

# ─── 7. Reconstruct & evaluate ─────────────────────────────────────
def reconstruct_and_evaluate(model_path: str, data: np.ndarray, attr_idx: list[int], batch_size):
    """Reconstruct selected attributes and compute reconstruction error.

    Parameters
    ----------
    model_path : str
        Path to the saved Keras autoencoder model.
    data : np.ndarray
        Array with shape (n_samples, n_timesteps, n_features) containing the
        original sequences.
    attr_idx : list[int]
        Indices of the attributes to evaluate.

    Returns
    -------
    dict
        Dictionary with MSE, MAE and RMSE for each selected attribute.
    np.ndarray
        Reconstructed data for the selected attributes.
    """
    # Load model
    model = tf.keras.models.load_model(
    model_path,
    compile=False,
    custom_objects={'r2': r2}     
)

    # Reconstruct
    ds = tf.data.Dataset.from_tensor_slices(data.astype(np.float32))
    ds = ds.batch(batch_size)
    recon = model.predict(ds, verbose=1)

    # Select requested attributes
    orig_subset = data[:, :, attr_idx]
    recon_subset = recon[:, :, attr_idx]

    # Compute metrics along samples and time dimensions
    mse = np.mean((orig_subset - recon_subset) ** 2, axis=(0, 1))
    mae = np.mean(np.abs(orig_subset - recon_subset), axis=(0, 1))
    rmse = np.sqrt(mse)

    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }
    return metrics, recon_subset


