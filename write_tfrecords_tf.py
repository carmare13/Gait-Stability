# 
import os, math
import numpy as np
import tensorflow as tf
from time import time


def write_sharded_tfrecord(
    npy_paths:  list[str],
    output_dir: str,
    shard_size: int
):
    """Shards sólo con 'data' (sin labels)."""
    os.makedirs(output_dir, exist_ok=True)
    total = sum(np.load(p).shape[0] for p in npy_paths)
    num_shards = math.ceil(total / shard_size)
    writer = None
    shard_idx = example_idx = 0
    start = time()
    for p in npy_paths:
        arr = np.load(p).astype(np.float32)
        for record in arr:
            if example_idx % shard_size == 0:
                if writer: writer.close()
                name = f"data-{shard_idx:05d}-of-{num_shards:05d}.tfrecord.gz"
                path = os.path.join(output_dir, name)
                opts = tf.io.TFRecordOptions(compression_type="GZIP")
                writer = tf.io.TFRecordWriter(path, opts)
                shard_idx += 1
            feat = {
                "data": tf.train.Feature(
                           bytes_list=tf.train.BytesList(value=[record.tobytes()])
                )
            }
            ex = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(ex.SerializeToString())
            example_idx += 1
    if writer: writer.close()
    print(f"Shards de DATA escritos en '{output_dir}/'")

def write_sharded_tfrecord_wlabels(
    npy_paths:  list[str],
    output_dir: str,
    shard_size: int,
    split_idx:  int 
):
    """Shards con 'data' (0:split_idx) y 'label' (split_idx:)."""
    os.makedirs(output_dir, exist_ok=True)
    total = sum(np.load(p).shape[0] for p in npy_paths)
    num_shards = math.ceil(total / shard_size)
    writer = None
    shard_idx = example_idx = 0
    start = time()
    for p in npy_paths:
        arr = np.load(p).astype(np.float32)
        for record in arr:
            if example_idx % shard_size == 0:
                if writer: writer.close()
                name = f"sup-{shard_idx:05d}-of-{num_shards:05d}.tfrecord.gz"
                path = os.path.join(output_dir, name)
                opts = tf.io.TFRecordOptions(compression_type="GZIP")
                writer = tf.io.TFRecordWriter(path, opts)
                shard_idx += 1
            data  = record[:, :split_idx]
            label = record[:, split_idx:]
            feat = {
                "data":  tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()])),
                "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
            }
            ex = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(ex.SerializeToString())
            example_idx += 1
    if writer: writer.close()
    print(f"Shards supervisados escritos en '{output_dir}/'")