"""
AE_pipeline_pytorch.py

Módulo base con:
- Configuración de dispositivo y semillas
- Dataset GaitBatchIterable para Zarr
- Modelos (AE LSTM, BiLSTM, Conv, SemiSupAE, EMATeacher, etc.)
- Funciones genéricas de entrenamiento y evaluación

Los experimentos concretos (train_*.py) deben importar desde aquí.
"""
# ─── Imports ─────────────────────────────────────────────────────────────
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
import zarr
from torch.utils.data import IterableDataset, get_worker_info
import torch.nn.functional as F
from pathlib import Path
import time
import psutil
from sklearn.metrics import mean_absolute_error, r2_score
import json 
# ─── Seeds ─────────────────────────────────────────────────────────────
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    try:
        torch.cuda.manual_seed_all(42)
    except Exception:
        pass

# ─── Configuración de cuDNN ─────────────────────────────────────────────
# Garantizar reproducibilidad en cuDNN
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Configuración ──────────────────────────────────────────────────────
n_timesteps= 100 #cycle is normalized to 100 points

# ─── Device ────────────────────────────────────────────────────────────
def get_device():
    # Si exportas USE_GPU=0, vas en CPU aunque haya GPU
    use_gpu = os.environ.get("USE_GPU", "1") == "1"
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")



# ─── 1. Dataset ─────────────────────────────────────────────────────────
class GaitBatchIterable(IterableDataset):
    def __init__(self, store_path, batch_size, return_meta=False,
                 shuffle=False, seed=42):    
        base = Path(__file__).resolve().parent  # carpeta del AE_pipeline_pytorch.py
        p = Path(store_path)
        self.store_path = store_path
        self.bs = batch_size
        self.return_meta = return_meta
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        import os, numpy as np, torch, zarr
        from torch.utils.data import get_worker_info

        # Limitar hilos (mitiga rarezas de blosc/OpenMP)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            import numcodecs.blosc as nblosc
            nblosc.set_nthreads(1)
        except Exception:
            pass

        # Abrir Zarr (consolidado si existe)
        try:
            grp = zarr.open_consolidated(self.store_path, mode="r")
            z = grp["data"]
        except Exception:
            z = zarr.open(self.store_path, mode="r")["data"]

        n = int(z.shape[0])

        worker_info = get_worker_info()
        base_seed = self.seed
        if worker_info:
            base_seed = self.seed + worker_info.id
        rng = np.random.default_rng(base_seed)


        batch_starts = np.arange(0, n, self.bs, dtype=np.int64)
        if self.shuffle:
           rng.shuffle(batch_starts)

        if worker_info:
            wid, nw = worker_info.id, worker_info.num_workers
            total = len(batch_starts)
            per, rem = total // nw, total % nw
            start = wid * per + min(wid, rem)
            end = start + per + (1 if wid < rem else 0)
            batch_starts = batch_starts[start:end]

        for s in batch_starts:
            lo = int(s)
            hi = int(min(s + self.bs, n))

            # Slicing contiguo (evita fancy indexing)
            data = z[lo:hi].astype("float32", copy=True)   # COPIA SEGURA

            # Split features / meta
            feat_np = data[:, :, :321]
            feat = torch.tensor(feat_np, dtype=torch.float32)
            cycle_id = torch.arange(lo, hi, dtype=torch.int64) # ID de ciclo global (útil para evitar fugas entre train/val)
            if self.return_meta:
                meta_np = data[:, 0, 321:326]
                # Conversión con copia explícita a tensor (más seguro que from_numpy)
                feat = torch.tensor(feat_np, dtype=torch.float32)
                meta = torch.tensor(meta_np, dtype=torch.float32)
                yield feat, meta, cycle_id
            else:
                feat = torch.tensor(feat_np, dtype=torch.float32)
                yield feat, feat, cycle_id


# This adds epoch-based shuffling by incorporating the epoch number into the RNG seed.
# class GaitBatchIterable(IterableDataset):
#     def __init__(self, store_path, batch_size, return_meta=False,
#                  shuffle=False, seed=42):
#         self.store_path = store_path
#         self.bs = batch_size
#         self.return_meta = return_meta
#         self.shuffle = shuffle
#         self.seed = seed
#         self.epoch = 0  # <- NEW

#     def set_epoch(self, epoch: int):
#         """Call this at the start of each epoch (training only)."""
#         self.epoch = int(epoch)

#     def __iter__(self):
#         import os, numpy as np, torch, zarr
#         from torch.utils.data import get_worker_info

#         os.environ.setdefault("OMP_NUM_THREADS", "1")
#         try:
#             import numcodecs.blosc as nblosc
#             nblosc.set_nthreads(1)
#         except Exception:
#             pass

#         try:
#             grp = zarr.open_consolidated(self.store_path, mode="r")
#             z = grp["data"]
#         except Exception:
#             z = zarr.open(self.store_path, mode="r")["data"]

#         n = int(z.shape[0])

#         worker_info = get_worker_info()
#         wid = worker_info.id if worker_info else 0

#         # NEW: epoch enters the RNG seed so shuffle changes each epoch but is reproducible
#         # Use a large multiplier to avoid collisions across epochs.
#         base_seed = int(self.seed) + 100_000 * int(self.epoch) + int(wid)
#         rng = np.random.default_rng(base_seed)

#         batch_starts = np.arange(0, n, self.bs, dtype=np.int64)
#         if self.shuffle:
#             rng.shuffle(batch_starts)

#         # Deterministic split across workers
#         if worker_info:
#             nw = worker_info.num_workers
#             total = len(batch_starts)
#             per, rem = total // nw, total % nw
#             start = wid * per + min(wid, rem)
#             end = start + per + (1 if wid < rem else 0)
#             batch_starts = batch_starts[start:end]

#         for s in batch_starts:
#             lo = int(s)
#             hi = int(min(s + self.bs, n))

#             data = z[lo:hi].astype("float32", copy=True)

#             feat_np = data[:, :, :321]
#             feat = torch.tensor(feat_np, dtype=torch.float32)

#             if self.return_meta:
#                 meta_np = data[:, :, 321:]
#                 meta = torch.tensor(meta_np, dtype=torch.float32)
#                 yield feat, meta
#             else:
#                 yield feat, feat                
# USE 
# train_ds = GaitBatchIterable(train_path, micro_batch, return_meta=False, shuffle=True, seed=42)
# train_loader = DataLoader(train_ds, batch_size=None, num_workers=num_workers, persistent_workers=False)

# for epoch in range(num_epochs):
#     train_ds.set_epoch(epoch)   # <- clave
#     for xb, yb in train_loader:
#         ...

class MultiSubjectGaitBatchIterable2(IterableDataset):
    """
    Multi-subject batcher for Zarr store shaped like:
      data [N, T=100, C=feat_dim + meta_dim]
    where meta columns are stored after the features along the last dimension.

    It samples `patients_per_batch` subjects and `samples_per_patient` cycles per subject.

    Yields:
      feat:     [B, T, feat_dim] float32
      meta0:    [B, meta_dim]   float32 (from t=0 only) as stored in zarr
      cycle_id: [B] int64 (global cycle indices)
    """

    def __init__(
        self,
        store_path,
        subject_slices_json,
        patients_per_batch=16,
        samples_per_patient=16,
        return_meta=True,
        shuffle_subjects=True,
        seed=42,
        feat_dim=321,
        t_steps=100,
        meta_dim=5,
        infinite=True,
        require_full_batch=True,

        # --- sampling options ---
        no_replace_within_subject=True,
        trial_slices_json=None,          # sid -> trial -> [lo,hi)
        trial_balanced=True,
        trial_day_slices_json=None,      # sid -> trial -> day -> [lo,hi)
        day_balanced=True,
        use_bin_sampling=True,           # for trial-day balancing (fast + diverse)
        sort_indices_for_io=True,
    ):
        super().__init__()
        self.store_path = str(store_path)
        self.return_meta = bool(return_meta)

        self.patients_per_batch = int(patients_per_batch)
        self.samples_per_patient = int(samples_per_patient)
        self.bs = self.patients_per_batch * self.samples_per_patient

        self.shuffle_subjects = bool(shuffle_subjects)
        self.seed = int(seed)
        self.epoch = 0

        self.feat_dim = int(feat_dim)
        self.meta_dim = int(meta_dim)
        self.t_steps = int(t_steps)

        self.infinite = bool(infinite)
        self.require_full_batch = bool(require_full_batch)

        self.no_replace_within_subject = bool(no_replace_within_subject)
        self.trial_balanced = bool(trial_balanced)
        self.day_balanced = bool(day_balanced)
        self.use_bin_sampling = bool(use_bin_sampling)
        self.sort_indices_for_io = bool(sort_indices_for_io)

        # subject -> [lo, hi)
        obj = json.loads(Path(subject_slices_json).read_text())
        self.slices = {int(k): (int(v[0]), int(v[1])) for k, v in obj.items()}
        self.subjects = sorted(self.slices.keys())

        # trial slices (optional)
        self.trial_slices = None
        if trial_slices_json is not None:
            obj_t = json.loads(Path(trial_slices_json).read_text())
            ts = {}
            for sid_str, d in obj_t.items():
                sid = int(sid_str)
                ts[sid] = {int(tr_str): (int(v[0]), int(v[1])) for tr_str, v in d.items()}
            self.trial_slices = ts

        # trial-day slices (optional)  sid -> trial -> day -> (lo,hi)
        self.trial_day_slices = None
        if trial_day_slices_json is not None:
            obj_td = json.loads(Path(trial_day_slices_json).read_text())
            td = {}
            for sid_str, tr_dict in obj_td.items():
                sid = int(sid_str)
                td[sid] = {}
                for tr_str, day_dict in tr_dict.items():
                    tr = int(tr_str)
                    td[sid][tr] = {int(day): (int(v[0]), int(v[1])) for day, v in day_dict.items()}
            self.trial_day_slices = td

        # --- per-worker state for no-replace ---
        self._perm_subj = None  # dict[sid] -> perm of local indices [0..n-1]
        self._ptr_subj = None   # dict[sid] -> pointer
        self._perm_trial = None # dict[(sid, trial)] -> perm
        self._ptr_trial = None  # dict[(sid, trial)] -> pointer

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    @staticmethod
    def _set_thread_safety():
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            import numcodecs.blosc as nblosc
            nblosc.set_nthreads(1)
        except Exception:
            pass

    def _open_zarr(self):
        try:
            grp = zarr.open_consolidated(self.store_path, mode="r")
            z = grp["data"]
        except Exception:
            z = zarr.open(self.store_path, mode="r")["data"]
        return z

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _sample_bins(lo, hi, k, rng):
        """Pick k indices spread over [lo,hi) using k bins (diverse, simple)."""
        n = hi - lo
        if n <= 0:
            return None
        if n <= k:
            base = np.arange(lo, hi, dtype=np.int64)
            if base.size < k:
                extra = rng.integers(lo, hi, size=(k - base.size), dtype=np.int64)
                base = np.concatenate([base, extra])
            return base

        edges = np.linspace(lo, hi, num=k + 1, dtype=np.int64)
        out = np.empty(k, dtype=np.int64)
        for i in range(k):
            a = int(edges[i])
            b = int(edges[i + 1])
            if b <= a:
                b = min(a + 1, hi)
            out[i] = int(rng.integers(a, b))
        return out

    # -----------------------------
    # No-replacement state init
    # -----------------------------
    def _init_subject_state_if_needed(self, rng):
        if (self._perm_subj is not None) and (self._ptr_subj is not None):
            return
        self._perm_subj = {}
        self._ptr_subj = {}
        for sid in self.subjects:
            lo, hi = self.slices[sid]
            n = hi - lo
            if n > 0:
                self._perm_subj[sid] = rng.permutation(n).astype(np.int64)
                self._ptr_subj[sid] = 0

    def _init_trial_state_if_needed(self, rng):
        if self.trial_slices is None:
            return
        if (self._perm_trial is not None) and (self._ptr_trial is not None):
            return
        self._perm_trial = {}
        self._ptr_trial = {}
        for sid, trials in self.trial_slices.items():
            for tr, (lo, hi) in trials.items():
                n = hi - lo
                if n > 0:
                    self._perm_trial[(sid, tr)] = rng.permutation(n).astype(np.int64)
                    self._ptr_trial[(sid, tr)] = 0

    def _sample_no_replace_range(self, perm, ptr, lo, hi, k, rng):
        n = hi - lo
        if n <= 0:
            return None, perm, ptr

        if n < k:
            base = lo + np.arange(n, dtype=np.int64)
            extra = rng.integers(lo, hi, size=(k - n), dtype=np.int64)
            return np.concatenate([base, extra]), perm, ptr

        if ptr + k <= n:
            idx_local = perm[ptr:ptr + k]
            ptr = ptr + k
            return lo + idx_local, perm, ptr

        first = perm[ptr:]
        perm2 = rng.permutation(n).astype(np.int64)
        need = k - (n - ptr)
        second = perm2[:need]
        perm = perm2
        ptr = need
        return lo + np.concatenate([first, second]), perm, ptr

    def _sample_subject_no_replace(self, sid, k, rng):
        lo, hi = self.slices[sid]
        perm = self._perm_subj.get(sid, None)
        ptr = self._ptr_subj.get(sid, 0)
        if perm is None:
            return None
        picks, perm, ptr = self._sample_no_replace_range(perm, ptr, lo, hi, k, rng)
        self._perm_subj[sid] = perm
        self._ptr_subj[sid] = ptr
        return picks

    def _sample_subject_trials_no_replace(self, sid, k, rng):
        """Your previous trial-balanced no-replace sampler (kept)."""
        trials = self.trial_slices.get(sid, None) if self.trial_slices is not None else None
        if not trials:
            return None

        trial_ids = [tr for tr in sorted(trials.keys()) if (trials[tr][1] - trials[tr][0]) > 0]
        if not trial_ids:
            return None

        picks_all = []

        if not self.trial_balanced:
            remaining = k
            while remaining > 0 and trial_ids:
                tr = int(rng.choice(trial_ids))
                lo, hi = trials[tr]
                key = (sid, tr)
                perm = self._perm_trial.get(key, None)
                if perm is None:
                    n = hi - lo
                    if n <= 0:
                        trial_ids.remove(tr)
                        continue
                    perm = rng.permutation(n).astype(np.int64)
                    self._perm_trial[key] = perm
                    self._ptr_trial[key] = 0

                ptr = self._ptr_trial[key]
                got, perm2, ptr2 = self._sample_no_replace_range(perm, ptr, lo, hi, remaining, rng)
                if got is None or got.size == 0:
                    trial_ids.remove(tr)
                    continue
                self._perm_trial[key] = perm2
                self._ptr_trial[key] = ptr2
                picks_all.append(got)
                remaining -= got.size
        else:
            ntr = len(trial_ids)
            base = k // ntr
            rem = k % ntr
            trial_order = trial_ids.copy()
            rng.shuffle(trial_order)

            for tr in trial_order:
                need = base + (1 if rem > 0 else 0)
                if rem > 0:
                    rem -= 1
                if need <= 0:
                    continue

                lo, hi = trials[tr]
                key = (sid, tr)
                perm = self._perm_trial.get(key, None)
                if perm is None:
                    n = hi - lo
                    if n <= 0:
                        continue
                    perm = rng.permutation(n).astype(np.int64)
                    self._perm_trial[key] = perm
                    self._ptr_trial[key] = 0

                ptr = self._ptr_trial[key]
                got, perm2, ptr2 = self._sample_no_replace_range(perm, ptr, lo, hi, need, rng)
                if got is None or got.size == 0:
                    continue
                self._perm_trial[key] = perm2
                self._ptr_trial[key] = ptr2
                picks_all.append(got)

            total_got = int(sum(x.size for x in picks_all))
            remaining = k - total_got
            if remaining > 0:
                extra = self._sample_subject_no_replace(sid, remaining, rng)
                if extra is not None and extra.size > 0:
                    picks_all.append(extra)

        if not picks_all:
            return None
        picks = np.concatenate(picks_all).astype(np.int64)
        if picks.size > k:
            picks = picks[:k]
        return picks

    def _sample_subject_trial_day_balanced_bins(self, sid, k, rng):
        """
        Uses trial_day_slices to spread picks across days (and trials).
        This is SIMPLE and does not need pointer-state. Great for val/test determinism + diversity.
        """
        if self.trial_day_slices is None or sid not in self.trial_day_slices:
            return None

        # pools_by_day: day -> list[(lo,hi)] across trials
        pools_by_day = {}
        for tr, day_dict in self.trial_day_slices[sid].items():
            for day, (lo, hi) in day_dict.items():
                if (hi - lo) <= 0:
                    continue
                pools_by_day.setdefault(day, []).append((lo, hi))

        days = sorted(pools_by_day.keys())
        if not days:
            return None

        # allocate roughly evenly over days (handles your 1 subject missing day2 automatically)
        base = k // len(days)
        rem = k % len(days)

        day_order = days.copy()
        rng.shuffle(day_order)

        picks = []
        for d in day_order:
            need = base + (1 if rem > 0 else 0)
            if rem > 0:
                rem -= 1
            if need <= 0:
                continue

            pools = pools_by_day[d]
            rng.shuffle(pools)

            # distribute "need" across pools
            per_pool = max(1, need // len(pools))
            left = need

            for (lo, hi) in pools:
                take = min(per_pool, left)
                got = self._sample_bins(lo, hi, take, rng)
                if got is not None and got.size > 0:
                    picks.append(got)
                    left -= got.size
                if left <= 0:
                    break

            # top-up if still missing
            while left > 0:
                lo, hi = pools[int(rng.integers(0, len(pools)))]
                got = self._sample_bins(lo, hi, 1, rng)
                if got is not None:
                    picks.append(got)
                    left -= 1

        out = np.concatenate(picks).astype(np.int64)
        if out.size > k:
            out = out[:k]
        return out

    # -----------------------------
    # Main iteration
    # -----------------------------
    def _iter_once(self, z, rng, worker_info):
        subjects = self.subjects.copy()
        if self.shuffle_subjects:
            rng.shuffle(subjects)

        batches = [
            subjects[j:j + self.patients_per_batch]
            for j in range(0, len(subjects) - self.patients_per_batch + 1, self.patients_per_batch)
        ]

        if worker_info:
            wid = worker_info.id
            nw = worker_info.num_workers
            batches = batches[wid::nw]

        # torch generator for deterministic shuffles (esp. val/test)
        # tied to seed+epoch+worker
        gen = torch.Generator()
        gen.manual_seed(int(self.seed + 100_000 * self.epoch + (worker_info.id if worker_info else 0)))

        for batch_subj in batches:
            idxs_list = []

            for sid in batch_subj:
                lo, hi = self.slices[sid]
                n_cycles = hi - lo
                if n_cycles <= 0:
                    if self.require_full_batch:
                        idxs_list = []
                        break
                    else:
                        continue

                if self.no_replace_within_subject:
                    # Priority: trial-day balancing (if provided)
                    if (self.trial_day_slices is not None) and self.day_balanced and self.use_bin_sampling:
                        picks = self._sample_subject_trial_day_balanced_bins(sid, self.samples_per_patient, rng)
                        if picks is None:
                            # fallback: trial-level no-replace if available
                            if self.trial_slices is not None:
                                picks = self._sample_subject_trials_no_replace(sid, self.samples_per_patient, rng)
                            else:
                                picks = self._sample_subject_no_replace(sid, self.samples_per_patient, rng)

                    # Else: trial-level no-replace if trial_slices exists
                    elif self.trial_slices is not None:
                        picks = self._sample_subject_trials_no_replace(sid, self.samples_per_patient, rng)
                        if picks is None:
                            picks = self._sample_subject_no_replace(sid, self.samples_per_patient, rng)

                    # Else: subject-level no-replace
                    else:
                        picks = self._sample_subject_no_replace(sid, self.samples_per_patient, rng)

                else:
                    # Old behavior: random window / replacement
                    if n_cycles >= self.samples_per_patient:
                        start = rng.integers(lo, hi - self.samples_per_patient + 1)
                        picks = np.arange(start, start + self.samples_per_patient, dtype=np.int64)
                    else:
                        picks = rng.integers(lo, hi, size=self.samples_per_patient, dtype=np.int64)

                if picks is None or picks.size == 0:
                    if self.require_full_batch:
                        idxs_list = []
                        break
                    else:
                        continue

                idxs_list.append(picks)

            if not idxs_list:
                continue

            idxs = np.concatenate(idxs_list).astype(np.int64)
            if self.require_full_batch and idxs.shape[0] != self.bs:
                continue

            # Sort indices for IO
            if self.sort_indices_for_io:
                order = np.argsort(idxs)
                idxs_read = idxs[order]
            else:
                idxs_read = idxs

            data = z.get_orthogonal_selection((idxs_read, slice(None), slice(None)))
            if data.dtype != np.float32:
                data = data.astype("float32", copy=False)

            feat = torch.from_numpy(data[:, :, :self.feat_dim]).contiguous()

            meta0 = None
            if self.return_meta:
                meta0 = torch.from_numpy(
                    data[:, 0, self.feat_dim:self.feat_dim + self.meta_dim]
                ).contiguous()

            cycle_id = torch.from_numpy(idxs_read).to(torch.int64)

            # Shuffle after loading
            perm = torch.randperm(feat.shape[0], generator=gen)
            feat = feat[perm]
            cycle_id = cycle_id[perm]
            if meta0 is not None:
                meta0 = meta0[perm]

            if self.return_meta:
                yield feat, meta0, cycle_id
            else:
                yield feat, feat, cycle_id

    def __iter__(self):
        self._set_thread_safety()
        z = self._open_zarr()

        worker_info = get_worker_info()
        wid = worker_info.id if worker_info else 0

        rng = np.random.default_rng(self.seed + 100_000 * self.epoch + wid)

        # init pointer-state only needed for no-replace samplers (subject/trial)
        if self.no_replace_within_subject:
            self._init_subject_state_if_needed(rng)
            self._init_trial_state_if_needed(rng)

        if self.infinite:
            while True:
                yield from self._iter_once(z, rng, worker_info)
                self.epoch += 1
                rng = np.random.default_rng(self.seed + 100_000 * self.epoch + wid)
        else:
            yield from self._iter_once(z, rng, worker_info)






class MultiSubjectGaitBatchIterable(IterableDataset):
    """
    Multi-subject batcher for Zarr store shaped like:
      data [N, T=100, C=feat_dim + meta_dim]
    where meta columns are stored after the features along the last dimension.

    It samples `patients_per_batch` subjects and `samples_per_patient` cycles per subject.

    Yields:
      feat:     [B, 100, feat_dim] float32
      meta0:    [B, meta_dim]      (from t=0 only) float32 (as stored)
      cycle_id: [B] int64 (global cycle indices)
    """

    def __init__(
        self,
        store_path,
        subject_slices_json,
        patients_per_batch=16,
        samples_per_patient=16,
        return_meta=True,
        shuffle_subjects=True,
        seed=42,
        feat_dim=321,
        t_steps=100,
        meta_dim=5,
        infinite=True,
        require_full_batch=True,
    ):
        super().__init__()
        self.store_path = str(store_path)
        self.return_meta = bool(return_meta)

        self.patients_per_batch = int(patients_per_batch)
        self.samples_per_patient = int(samples_per_patient)
        self.bs = self.patients_per_batch * self.samples_per_patient

        self.shuffle_subjects = bool(shuffle_subjects)
        self.seed = int(seed)
        self.epoch = 0

        self.feat_dim = int(feat_dim)
        self.meta_dim = int(meta_dim)
        self.t_steps = int(t_steps)

        self.infinite = bool(infinite)
        self.require_full_batch = bool(require_full_batch)

        # subject -> [lo, hi)
        obj = json.loads(Path(subject_slices_json).read_text())
        self.slices = {int(k): (int(v[0]), int(v[1])) for k, v in obj.items()}
        self.subjects = sorted(self.slices.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    @staticmethod
    def _set_thread_safety():
        # Mitigate OpenMP / blosc thread issues
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            import numcodecs.blosc as nblosc
            nblosc.set_nthreads(1)
        except Exception:
            pass

    def _open_zarr(self):
        # Open zarr (once per worker process)
        try:
            grp = zarr.open_consolidated(self.store_path, mode="r")
            z = grp["data"]
        except Exception:
            z = zarr.open(self.store_path, mode="r")["data"]
        return z

    def _iter_once(self, z, rng, worker_info):
        # Prepare subject list for this pass
        subjects = self.subjects.copy()
        if self.shuffle_subjects:
            rng.shuffle(subjects)

        # Build batches of subjects
        batches = [
            subjects[j:j + self.patients_per_batch]
            for j in range(0, len(subjects) - self.patients_per_batch + 1, self.patients_per_batch)
        ]

        # Split batches across workers (safe)
        if worker_info:
            wid = worker_info.id
            nw = worker_info.num_workers
            batches = batches[wid::nw]

        for batch_subj in batches:
            idxs_list = []
            for sid in batch_subj:
                lo, hi = self.slices[sid]
                n_cycles = hi - lo
                if n_cycles <= 0:
                    if self.require_full_batch:
                        idxs_list = []
                        break
                    else:
                        continue

                if n_cycles >= self.samples_per_patient:
                    start = rng.integers(lo, hi - self.samples_per_patient + 1)
                    picks = np.arange(start, start + self.samples_per_patient, dtype=np.int64)
                else:
                    # Fallback: sample with replacement
                    picks = rng.integers(lo, hi, size=self.samples_per_patient, dtype=np.int64)

                idxs_list.append(picks)

            if not idxs_list:
                continue

            idxs = np.concatenate(idxs_list).astype(np.int64)  # [B']
            if self.require_full_batch and idxs.shape[0] != self.bs:
                continue

            # Single read per batch
            data = z.get_orthogonal_selection((idxs, slice(None), slice(None)))
            if data.dtype != np.float32:
                data = data.astype("float32", copy=False)

            # Features
            feat = torch.from_numpy(data[:, :, :self.feat_dim]).contiguous()

            # Meta only at t=0 -> [B, meta_dim]
            meta0 = None
            if self.return_meta:
                meta0 = torch.from_numpy(
                    data[:, 0, self.feat_dim:self.feat_dim + self.meta_dim]
                ).contiguous()

            # Cycle id (global indices)
            cycle_id = torch.from_numpy(idxs).to(torch.int64)

            # Shuffle AFTER loading (cheap)
            perm = torch.randperm(feat.shape[0])
            feat = feat[perm]
            cycle_id = cycle_id[perm]
            if meta0 is not None:
                meta0 = meta0[perm]

            if self.return_meta:
                yield feat, meta0, cycle_id
            else:
                # Compatibility: (x, y, cycle_id)
                yield feat, feat, cycle_id

    def __iter__(self):
        self._set_thread_safety()
        z = self._open_zarr()

        worker_info = get_worker_info()

        # Worker-aware RNG
        wid = worker_info.id if worker_info else 0
        rng = np.random.default_rng(self.seed + 100_000 * self.epoch + wid)

        # Infinite stream or single pass
        if self.infinite:
            while True:
                yield from self._iter_once(z, rng, worker_info)
                self.epoch += 1
                # re-seed each epoch for reproducible reshuffling
                rng = np.random.default_rng(self.seed + 100_000 * self.epoch + wid)
        else:
            yield from self._iter_once(z, rng, worker_info)


class MultiSubjectGaitBatchIterable_XMeta(IterableDataset):
    """
    Zarr store must contain:
      X    [N, 100, 321] float32
      meta [N, 5] int32  (subject, group, day, block, trial)

    Batching:
      - patients_per_batch subjects per batch
      - samples_per_patient cycles per subject
      - total batch size B = patients_per_batch * samples_per_patient

    Returns:
      x:        [B,100,321] float32
      meta0:    [B,5]       int32
      cycle_id: [B]         int64 (global cycle indices)
    """

    def __init__(
        self,
        store_path,
        subject_slices_json,
        patients_per_batch=16,
        samples_per_patient=16,
        shuffle_subjects=True,
        seed=42,
        require_full_batch=True,  # if True, drop any subject with empty slice and require B exactly
    ):
        super().__init__()
        self.store_path = str(store_path)
        self.patients_per_batch = int(patients_per_batch)
        self.samples_per_patient = int(samples_per_patient)
        self.bs = self.patients_per_batch * self.samples_per_patient

        self.shuffle_subjects = bool(shuffle_subjects)
        self.seed = int(seed)
        self.epoch = 0
        self.require_full_batch = bool(require_full_batch)

        obj = json.loads(Path(subject_slices_json).read_text())
        self.slices = {int(k): (int(v[0]), int(v[1])) for k, v in obj.items()}
        self.subjects = sorted(self.slices.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _set_thread_safety(self):
        # Helps avoid oversubscription (important on clusters)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            import numcodecs.blosc as nblosc
            nblosc.set_nthreads(1)
        except Exception:
            pass

    def __iter__(self):
        self._set_thread_safety()

        # Open store (once per worker process)
        try:
            grp = zarr.open_consolidated(self.store_path, mode="r")
        except Exception:
            grp = zarr.open(self.store_path, mode="r")

        X = grp["X"]
        M = grp["meta"]

        worker_info = get_worker_info()
        wid = worker_info.id if worker_info else 0
        nw = worker_info.num_workers if worker_info else 1

        # Epoch-aware RNG seed (reproducible across epochs)
        rng = np.random.default_rng(self.seed + 100_000 * self.epoch + wid)

        subjects = self.subjects.copy()
        if self.shuffle_subjects:
            rng.shuffle(subjects)

        # ---- IMPORTANT CHANGE ----
        # Do NOT split subjects across workers. Split BATCHES across workers (strided),
        # so we never end up with <patients_per_batch subjects per worker.
        batches = [
            subjects[j:j + self.patients_per_batch]
            for j in range(0, len(subjects) - self.patients_per_batch + 1, self.patients_per_batch)
        ]
        if worker_info:
            batches = batches[wid::nw]

        # Iterate batches
        for batch_subj in batches:
            # Build idxs as blocks per subject (contiguous within each subject)
            idxs_list = []
            for sid in batch_subj:
                lo, hi = self.slices[sid]
                n = hi - lo
                if n <= 0:
                    if self.require_full_batch:
                        idxs_list = []
                        break
                    else:
                        continue

                if n >= self.samples_per_patient:
                    start0 = rng.integers(lo, hi - self.samples_per_patient + 1)
                    idxs = np.arange(start0, start0 + self.samples_per_patient, dtype=np.int64)
                else:
                    # sample with replacement
                    idxs = rng.integers(lo, hi, size=self.samples_per_patient, dtype=np.int64)

                idxs_list.append(idxs)

            if not idxs_list:
                continue

            idxs = np.concatenate(idxs_list).astype(np.int64)  # [B']
            if self.require_full_batch and idxs.shape[0] != self.bs:
                # ensure shape is stable if you rely on fixed batch size
                continue

            # ---- BIG WIN: only 2 reads per batch ----
            x_np = X.get_orthogonal_selection((idxs, slice(None), slice(None)))  # float32 already
            m_np = M.get_orthogonal_selection((idxs, slice(None)))              # int32 already

            # Convert once
            x = torch.from_numpy(x_np)       # [B,100,321]
            meta0 = torch.from_numpy(m_np)   # [B,5]
            cycle_id = torch.from_numpy(idxs).to(torch.int64)

            # Shuffle inside batch (after read)
            perm = torch.randperm(x.shape[0])
            yield x[perm].contiguous(), meta0[perm].contiguous(), cycle_id[perm].contiguous()




class MultiSubjectGaitBatchIterable_XMeta_Fast(IterableDataset):
    """
    Zarr store:
      X    [N, 100, 321] float32
      meta [N, 5] int32

    Returns:
      x        [B,100,321]
      meta0    [B,5]
      cycle_id [B]
    """
    def __init__(
        self,
        store_path,
        subject_slices_json,
        patients_per_batch=16,
        samples_per_patient=64,
        shuffle_subjects=True,
        seed=42,
        require_full_batch=True,
    ):
        super().__init__()
        self.store_path = str(store_path)
        self.patients_per_batch = int(patients_per_batch)
        self.samples_per_patient = int(samples_per_patient)
        self.bs = self.patients_per_batch * self.samples_per_patient
        self.shuffle_subjects = bool(shuffle_subjects)
        self.seed = int(seed)
        self.epoch = 0
        self.require_full_batch = bool(require_full_batch)

        obj = json.loads(Path(subject_slices_json).read_text())
        self.slices = {int(k): (int(v[0]), int(v[1])) for k, v in obj.items()}
        self.subjects = sorted(self.slices.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        # Avoid thread oversubscription on cluster
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            import numcodecs.blosc as nblosc
            nblosc.set_nthreads(1)
        except Exception:
            pass

        # Open once per worker
        try:
            grp = zarr.open_consolidated(self.store_path, mode="r")
        except Exception:
            grp = zarr.open(self.store_path, mode="r")

        X = grp["X"]
        M = grp["meta"]

        worker_info = get_worker_info()
        wid = worker_info.id if worker_info else 0
        nw  = worker_info.num_workers if worker_info else 1

        rng = np.random.default_rng(self.seed + 100_000 * self.epoch + wid)

        subjects = self.subjects.copy()
        if self.shuffle_subjects:
            rng.shuffle(subjects)

        # Build batches then stride by worker (prevents StopIteration from subject-splitting)
        batches = [
            subjects[j:j + self.patients_per_batch]
            for j in range(0, len(subjects) - self.patients_per_batch + 1, self.patients_per_batch)
        ]
        if worker_info:
            batches = batches[wid::nw]

        for batch_subj in batches:
            idxs_list = []
            for sid in batch_subj:
                lo, hi = self.slices[sid]
                n = hi - lo
                if n <= 0:
                    if self.require_full_batch:
                        idxs_list = []
                        break
                    else:
                        continue

                if n >= self.samples_per_patient:
                    start0 = rng.integers(lo, hi - self.samples_per_patient + 1)
                    idxs = np.arange(start0, start0 + self.samples_per_patient, dtype=np.int64)
                else:
                    # replacement
                    idxs = rng.integers(lo, hi, size=self.samples_per_patient, dtype=np.int64)

                idxs_list.append(idxs)

            if not idxs_list:
                continue

            idxs = np.concatenate(idxs_list).astype(np.int64)
            if self.require_full_batch and idxs.shape[0] != self.bs:
                continue

            # 2 reads per batch (fastest on network FS)
            x_np = X.get_orthogonal_selection((idxs, slice(None), slice(None)))  # float32
            m_np = M.get_orthogonal_selection((idxs, slice(None)))              # int32

            x = torch.from_numpy(x_np)
            meta0 = torch.from_numpy(m_np)
            cycle_id = torch.from_numpy(idxs).to(torch.int64)

            # shuffle after load
            perm = torch.randperm(x.shape[0])
            yield x[perm].contiguous(), meta0[perm].contiguous(), cycle_id[perm].contiguous()



# ---- Model's Blocks (shared by AE and SemiSupAE)-----------------
class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd): 
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): 
        return -ctx.lambd * grad_output, None

class GRL(nn.Module):
    def __init__(self, lambd=1.0): 
        super().__init__(); self.lambd = lambd
    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambd)
    

class BiLSTMEncoder(nn.Module):
    def __init__(self, in_dim=321, hidden=128, latent=128):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=in_dim, hidden_size=hidden,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.to_latent = nn.Sequential(
            nn.Linear(2*hidden, latent),
            nn.LayerNorm(latent)
        )
    def forward(self, x):  # x: [B, 100, 321]
        h, _ = self.bilstm(x)         # [B, 100, 256]
        h_last = h[:, -1, :]          # take last timestep  [B, 256]
        z = self.to_latent(h_last)    # [B, 128]
        return z

class LSTMDecoder(nn.Module):
    def __init__(self, out_dim=321, hidden=128, steps=100, latent=128):
        super().__init__()
        self.steps = steps
        self.init = nn.Linear(latent, hidden)
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden,
                            num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, z):  # z: [B, 128]
        B = z.size(0)
        h0 = torch.tanh(self.init(z))          # [B, 128]
        seq = h0.unsqueeze(1).repeat(1, self.steps, 1)  # teacher-free decoding
        y, _ = self.lstm(seq)                  # [B, 100, 128]
        y = self.out(y)                        # [B, 100, 321]
        return y

class HeadMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

# ---------- Models ----------
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

class SemiSupAE(nn.Module):
    def __init__(self, steps=100, in_dim=321, latent=128,
                 n_group=2, n_nuisance=None, n_subjects=None, grl_lambda=1.0):
        super().__init__()
        self.encoder = BiLSTMEncoder(in_dim=in_dim, latent=latent)
        self.decoder = LSTMDecoder(out_dim=in_dim, steps=steps, latent=latent)
        self.group_head = HeadMLP(latent, 64, n_group)
        self.proj_head  = ProjectionHead(latent, 128)
        
        # Nuisance Ej Day 
        self.use_nuis = n_nuisance is not None
        if self.use_nuis:
            self.grl = GRL(lambd=grl_lambda)
            self.nuis_head = HeadMLP(latent, 64, n_nuisance)

        # Adversarial para sujetos (Subject ID)
        self.use_subject_adv = n_subjects is not None
        if self.use_subject_adv:
            self.grl_subj = GRL(lambd=grl_lambda)
            self.subject_head = HeadMLP(latent, 128, n_subjects) # Clasificador de IDs    

    def forward(self, x, return_all=False):
        z = self.encoder(x)             # [B,128]
        y_hat = self.decoder(z)         # [B,100,321]
        logits = self.group_head(z)     # [B,2]
        proj = self.proj_head(z)        # [B,128]
        out = {"z": z, "recon": y_hat, "logits": logits, "proj": proj}
        if self.use_nuis:
            nuis_logits = self.nuis_head(self.grl(z))
            out["nuis_logits"] = nuis_logits
        if self.use_subject_adv:
            subj_logits = self.subject_head(self.grl_subj(z))
            out["subj_logits"] = subj_logits    
        return out if return_all else (y_hat, logits)

# ---------- EMA teacher for SemiSupAE ----------
class EMATeacher(nn.Module):
    def __init__(self, student, ema=0.99):
        super().__init__()
        import copy
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters(): p.requires_grad = False
        self.ema = ema
        self.update(student, init=True)
        for p in self.teacher.parameters(): p.requires_grad = False
    @torch.no_grad()
    def update(self, student, init=False):
        ts, ss = self.teacher.state_dict(), student.state_dict()
        for k in ts.keys():
            ts[k] = ss[k] if init else self.ema*ts[k] + (1-self.ema)*ss[k]
        self.teacher.load_state_dict(ts)

# ---------- Losses & Auxiliary functions ----------
def supcon_loss(emb, labels, temperature=0.07):
    # emb: [B, D] normalized; labels: [B] patient_id
    # Simple supervised contrastive (NT-Xent style)
    sim = torch.matmul(emb, emb.t()) / temperature
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.t()).float()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask
    # log-softmax over rows
    log_prob = sim - torch.logsumexp(sim + torch.log(logits_mask + 1e-12), dim=1, keepdim=True)
    mean_log_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    return -mean_log_pos.mean()

def consistency_loss(student_logits, teacher_logits):
    # MSE on probabilities (you can also use KL)
    ps = F.softmax(student_logits, dim=-1)
    pt = F.softmax(teacher_logits.detach(), dim=-1)
    return F.mse_loss(ps, pt)



# ---------- Training loop sketch ----------




# ─── 3. Training ────────────────────────────────────────────────────────
# def contrastive_loss(x1, x2, label, margin=1.0):
#     # Distancia euclidiana entre las representaciones latentes
#     euclidean_distance = F.pairwise_distance(x1, x2, keepdim=True)
#     loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
#                                   (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
#     return loss_contrastive

def train_step(batch, model, teacher, optimizer, weights,
               augment_fn, pseudo_thresh=0.9):
    """
    batch: dict with
      x: [B,100,321], y_group: [B] (may be -1 for unlabeled),
      patient_id: [B], nuis_label: [B] (optional)
    """
    model.train()
    x = batch['x']
    y_group = batch['y_group']          # -1 == unlabeled
    pid = batch['patient_id']
    nuis = batch.get('nuis', None)

    # Debug print for nuisance variable
    # if nuis is None:
    #     print("nuis is None")
    # else:
    #     print("nuis present, adv weight =", weights.get("adv", None))

    # Augment for consistency
    x_strong = augment_fn(x)
    out_s = model(x, return_all=True)
    if not hasattr(train_step, "first_check_done"):
        print(f"\n[DEBUG] subj_logits in out_s: {'subj_logits' in out_s}")
        train_step.first_check_done = True
    out_s_strong = model(x_strong, return_all=True)
    
    
    # Teacher forward (no grad)
    with torch.no_grad():
        out_t = teacher.teacher(x, return_all=True)

    # Loss: reconstruction
    L_rec = F.mse_loss(out_s['recon'], x)

    # Supervised group CE on labeled subset
    labeled_mask = (y_group >= 0)
    labeled_frac = labeled_mask.float().mean().item()
    #print(f"[train_step] Fraction of labeled samples in batch: {labeled_frac:.3f}")
    L_group = torch.tensor(0.0, device=x.device)
    if labeled_mask.any():
        L_group = F.cross_entropy(out_s['logits'][labeled_mask], y_group[labeled_mask], label_smoothing=0.05)

    # Adversarial per subject ID (if available
    L_subj_adv = torch.tensor(0.0, device=x.device)
    if model.use_subject_adv and weights.get("subj_adv", 0.0) > 0:
        # Aquí es donde el GRL actúa: el Encoder será penalizado si el subject_head acierta
        L_subj_adv = F.cross_entropy(out_s['subj_logits'], pid)#    

    # Pseudo-labels on unlabeled
    unl_mask = ~labeled_mask
    if unl_mask.any():
        probs = F.softmax(out_t['logits'][unl_mask], dim=-1)
        conf, pseudo = probs.max(-1)
        sel = conf >= pseudo_thresh
        if sel.any():
            L_pseudo = F.cross_entropy(out_s['logits'][unl_mask][sel], pseudo[sel])
            L_group = L_group + weights['pseudo_w'] * L_pseudo

    # Consistency (student strong vs teacher)
    L_cons = consistency_loss(out_s_strong['logits'], out_t['logits'])

    # SupCon with patient identity
    L_supcon = torch.tensor(0.0, device=x.device)
    if weights.get("supcon", 0.0) > 0:
        L_supcon = supcon_loss(out_s['proj'], pid)

    # Adversarial invariance to nuisance (if available)
    L_adv = torch.tensor(0.0, device=x.device)
    if weights.get("adv", 0.0) > 0 and nuis is not None and model.use_nuis:
        L_adv = F.cross_entropy(out_s['nuis_logits'], nuis)

    # Total
    L = (weights['rec']*L_rec + weights['group']*L_group + weights.get('subj_adv', 0.0) * L_subj_adv +
         weights['supcon']*L_supcon + weights['cons']*L_cons +
         weights['adv']*L_adv)

    optimizer.zero_grad()
    L.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    teacher.update(model)

    return { "loss": L.item(), "rec": L_rec.item(),
             "group": L_group.item() if isinstance(L_group, torch.Tensor) else L_group,
             "subj_adv": L_subj_adv.item() if isinstance(L_subj_adv, torch.Tensor) else L_subj_adv,
             "supcon": L_supcon.item(), "cons": L_cons.item(),
             "adv": L_adv.item() if isinstance(L_adv, torch.Tensor) else L_adv }





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


        if debug and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

        for step, (x, _) in enumerate(train_loader):
            if debug:
                t1 = time.perf_counter()

            x = x.to(device, non_blocking=True)

            if debug and device.type == 'cuda':
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

            if debug and device.type == 'cuda':
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
        if debug and device.type == 'cuda':
            epoch_gpu_peak = torch.cuda.max_memory_allocated(device)
            writer.add_scalar('Mem/GPU_peak_GB', epoch_gpu_peak/1e9, epoch)
            
        # CPU RSS 
        epoch_cpu_rss = process.memory_info().rss    
        writer.add_scalar('Mem/CPU_RSS_GB', epoch_cpu_rss/1e9, epoch)

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, _ in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    out_val = model(x_val)  # puede ser tensor o tupla (recon, z)
                recon_val = out_val[0] if isinstance(out_val, tuple) else out_val
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
            out = model(x_batch)
            recon = out[0] if isinstance(out, tuple) else out
            predictions.append(recon.cpu())
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
                out = model(x)
            recon = out[0] if isinstance(out, tuple) else out
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

