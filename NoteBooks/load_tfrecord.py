import numpy as np
import torch
from torch.utils.data import DataLoader
from tfrecord.torch.dataset import TFRecordDataset


BATCH_SIZE = 256
NUM_VARS = 321
n_timesteps= 100 

def _transform_data(record):
    flat = np.frombuffer(record["data"], dtype=np.float32)
    return torch.from_numpy(flat).view(-1, 321)

def create_data_dataloader(
    tf_paths:      list[str],
    idx_paths:     list[str],
    batch_size:    int = 128,
    shuffle:       bool= True,
    num_workers:   int = 0,
    pin_memory:    bool= False,
) -> DataLoader:
    ds = TFRecordDataset(
        data_path=tf_paths,
        index_path=idx_paths,
        description={"data": "byte"},
        num_epochs=1,
        shuffle_queue_size=100
    ).map(_transform_data)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=1, persistent_workers=False
    )

def _transform_supervised(record):
    flat_x = np.frombuffer(record["data"],  dtype=np.float32)
    flat_y = np.frombuffer(record["label"], dtype=np.float32)
    x = torch.from_numpy(flat_x).view(-1, 321)
    y = torch.from_numpy(flat_y).view(-1, flat_y.size // 100)  # ajusta según total_vars-321
    return x, y

def create_supervised_dataloader(
    tf_paths:      list[str],
    idx_paths:     list[str],
    batch_size:    int = 128,
    shuffle:       bool= False,
    num_workers:   int = 0,
    pin_memory:    bool= False,
) -> DataLoader:
    ds = TFRecordDataset(
        data_path=tf_paths,
        index_path=idx_paths,
        description={"data": "byte", "label": "byte"},
        num_epochs=1,
        shuffle_queue_size=100
    ).map(_transform_supervised)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=1, persistent_workers=False
    )