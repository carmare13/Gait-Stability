"""
Entrenamiento del SemiSupAE para la base de datos de marcha,
enfocado en obtener una buena representación por paciente.

Ejecutar desde terminal:

    conda activate gait-stability
    python train_semisupae_subject.py

Para correr en background:

    tmux new -s GaitAE
    cd /mnt/storage/dmartinez/Gait-Stability
    conda activate gait-stability
    python train_semisupae_subject.py > train_semisupae_subject.log 2>&1
"""

# === Config & seeds ===
import os
import random
import time
import json
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import zarr as _zarr

from AE_pipeline_pytorch import (
    GaitBatchIterable,
    SemiSupAE,
    EMATeacher,
    supcon_loss,
    consistency_loss,
    device
)

# -------------------------------------------------
#  Balanced train IterableDataset (by patient)
# -------------------------------------------------


class GaitBatchIterableBalanced(IterableDataset):
    """
    IterableDataset that yields patient-balanced batches.

    Each batch is built as:
        patients_per_batch * samples_per_patient = batch_size

    It does *random sampling with replacement* over cycles, but balances
    the number of samples per patient inside each batch.

    This dataset is conceptually "infinite", so the training loop must
    explicitly control the number of steps per epoch (we do that in main()).
    """

    def __init__(
        self,
        store_path: str,
        patients_per_batch: int = 16,
        samples_per_patient: int = 8,
        return_meta: bool = False,
    ):
        super().__init__()
        self.store_path = store_path
        self.ppb = patients_per_batch
        self.spp = samples_per_patient
        self.return_meta = return_meta

    def __iter__(self):
        import os
        import numpy as np
        import torch
        import zarr
        from torch.utils.data import get_worker_info

        # Limit threads (helps with blosc/OpenMP weirdness)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            import numcodecs.blosc as nblosc

            nblosc.set_nthreads(1)
        except Exception:
            pass

        # Open Zarr
        try:
            grp = zarr.open_consolidated(self.store_path, mode="r")
            z = grp["data"]
        except Exception:
            z = zarr.open(self.store_path, mode="r")["data"]

        n = len(z)

        # Worker info (each worker gets its own RNG)
        winfo = get_worker_info()
        rng = np.random.default_rng(winfo.id if winfo else None)

        # Pre-load meta to extract patient IDs
        # data shape expected: (N, T, 321 + meta_dim)
        meta_all = z[:, :, 321:]  # (N, T, meta_dim)
        # patient id assumed at meta[:, 0, 0]
        patient_ids = meta_all[:, 0, 0]  # (N,)

        unique_pats = np.unique(patient_ids)

        # Optionally split patients among workers
        if winfo:
            wid, nw = winfo.id, winfo.num_workers
            pats_split = np.array_split(unique_pats, nw)
            unique_pats = pats_split[wid]

        # Infinite stream of balanced batches.
        # Training loop will decide how many steps to take per epoch.
        while True:
            # 1) Choose patients for this batch
            if len(unique_pats) >= self.ppb:
                chosen_pats = rng.choice(unique_pats, size=self.ppb, replace=False)
            else:
                chosen_pats = rng.choice(unique_pats, size=self.ppb, replace=True)

            # 2) Build sample indices for this batch
            batch_indices = []
            for p in chosen_pats:
                idx_p = np.where(patient_ids == p)[0]
                if len(idx_p) == 0:
                    continue
                chosen_idx = rng.choice(idx_p, size=self.spp, replace=True)
                batch_indices.extend(chosen_idx.tolist())

            if not batch_indices:
                # In pathological case where no indices are chosen (should not happen),
                # just skip this iteration.
                continue

            batch_indices = np.array(batch_indices, dtype=np.int64)

            # 3) Fancy indexing to get data (safe copy)
            data = z.oindex[batch_indices, :, :]  # (B, T, 326)

            feat_np = data[:, :, :321].astype("float32")
            meta_np = data[:, :, 321:].astype("float32") if self.return_meta else None

            feat = torch.tensor(feat_np, dtype=torch.float32)

            if self.return_meta:
                meta = torch.tensor(meta_np, dtype=torch.float32)
                yield feat, meta
            else:
                yield feat, feat



# -------------------------------------------------
#  Seed & perf settings
# -------------------------------------------------

# Seeds
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Performance (if strict reproducibility is not required)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# (Ampere+): allow TF32 for GEMMs
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# Wrappers used in the loop
def loss_rec(out_recon, x):
    return F.mse_loss(out_recon, x)


def loss_group(logits, y):
    return F.cross_entropy(logits, y)


def loss_adv(nuis_logits, nuis):
    return F.cross_entropy(nuis_logits, nuis)


# === Misc utilities ===
def get_recon(out):
    # dict (semisup), tuple (AE) or tensor
    if isinstance(out, dict):
        return out["recon"]
    if isinstance(out, tuple):
        return out[0]
    return out


def fmt_hms(s):
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# === Model and training ===
def main():
    # ------------------------
    # Hyperparameters
    # ------------------------
    # Patient-balanced micro-batch:
    patients_per_batch = 8
    samples_per_patient = 4
    micro_batch = 32
    #micro_batch = patients_per_batch * samples_per_patient  # 16 * 8 = 128

    # Gradient accumulation:
    # 128 * 32 = 4096 effective samples per optimizer step
    accum_steps = 128

    num_workers = 8
    prefetch = 6

    num_epochs = 10
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16  # good for Ampere; can switch to float16 if needed

    print(
        f"patients_per_batch={patients_per_batch} | samples_per_patient={samples_per_patient} | "
        f"micro_batch={micro_batch} | accum_steps={accum_steps} | "
        f"workers={num_workers} | prefetch={prefetch}"
    )

    # ------------------------
    # Data paths
    # ------------------------
    train_path = str(Path("train_cycles.zarr").resolve())
    val_path = str(Path("val_cycles.zarr").resolve())
    test_path = str(Path("test_cycles.zarr").resolve())  # optional

    # Quick check on Zarr shapes
    p = Path(train_path)
    root = _zarr.open_group(str(p), mode="r")
    print("Arrays in train Zarr:", list(root.array_keys()))
    print("shape data train:", root["data"].shape)  # expected (N, 100, 326)
    train_n = root["data"].shape[0]

    # Decide how many train steps per epoch (approx one full pass over dataset)
    train_steps_per_epoch = 200
    #train_steps_per_epoch = max(1, train_n // micro_batch)
    print(f"train_n={train_n} | train_steps_per_epoch≈{train_steps_per_epoch}")

    # DataLoaders
    #  - Train: patient-balanced iterable
    #  - Val/Test: sequential iterable (contiguous slices)
    train_loader = DataLoader(
        GaitBatchIterableBalanced(
            train_path,
            patients_per_batch=patients_per_batch,
            samples_per_patient=samples_per_patient,
            return_meta=True,
        ),
        batch_size=None,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=None,
    )

    val_loader = DataLoader(
        GaitBatchIterable(val_path, batch_size=micro_batch, return_meta=True),
        batch_size=None,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
    )

    test_loader = DataLoader(
        GaitBatchIterable(test_path, batch_size=micro_batch, return_meta=True),
        batch_size=None,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
    )

    # One sample batch inspection
    x0, meta0 = next(iter(train_loader))
    print("First train batch x shape:", x0.shape)
    print("First train batch meta shape:", meta0.shape)

    # ------------------------
    # Model, EMA teacher, optimizer
    # ------------------------
    # Note: now 3 groups (young, middle, older)
    model = SemiSupAE(
        steps=100,
        in_dim=321,
        latent=128,
        n_group=3,
        n_nuisance=None,
    ).to(device)

    teacher = EMATeacher(model, ema=0.995)  # EMA-frozen teacher copy

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    min_lr = 1e-5
    scheduler = CosineAnnealingLR(opt, T_max=num_epochs, eta_min=min_lr)

    scaler = GradScaler()

    # Loss weights
    weights = dict(rec=1.0, group=0.5, supcon=2.0, cons=0.1, adv=0.0, pseudo_w=0.0)

    # (Optional) PyTorch 2.x: compile
    # model = torch.compile(model)
    # teacher.teacher = torch.compile(teacher.teacher)

    # ------------------------
    # Checkpointing
    # ------------------------
    CKPT_DIR = Path("checkpoints_3groups_subject")
    CKPT_DIR.mkdir(exist_ok=True)
    CKPT_GLOBAL = CKPT_DIR / "semisupae_subject_last.pt"

    def save_ckpt(step, path):
        torch.save(
            {
                "model": model.state_dict(),
                "teacher": teacher.teacher.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "micro_batch": micro_batch,
                "accum_steps": accum_steps,
            },
            path,
        )
        print(f"[ckpt] saved at {path} (step={step})")

    def load_ckpt(path):
        if not path.exists():
            print("[ckpt] not found, starting from scratch")
            return 0
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        teacher.teacher.load_state_dict(ckpt["teacher"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        print(f"[ckpt] loaded from {path} (step={ckpt.get('step', 0)})")
        return ckpt.get("step", 0)

    global_step = load_ckpt(CKPT_GLOBAL)

    # ------------------------
    # Epoch-wise logging (CSV + JSONL)
    # ------------------------
    RUN_ROOT = Path("runs_subject")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    RUN_NAME = "semisup_3groups_subject"  # change suffix for configs
    RUN_DIR = RUN_ROOT / f"{RUN_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    CSV_PATH = RUN_DIR / "metrics.csv"
    JSONL_PATH = RUN_DIR / "metrics.jsonl"
    FIELDS = [
        "epoch",
        "global_step",
        "train_loss",
        "train_rec",
        "val_loss",
        "val_ce",
        "val_acc",
        "seconds",
    ]

    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log_epoch_row(
        epoch,
        global_step,
        train_loss,
        train_rec,
        val_loss,
        val_ce,
        val_acc,
        seconds,
    ):
        row = dict(
            epoch=int(epoch),
            global_step=int(global_step),
            train_loss=float(train_loss),
            train_rec=float(train_rec),
            val_loss=float(val_loss),
            val_ce=float(val_ce),
            val_acc=float(val_acc),
            seconds=float(seconds),
        )
        # CSV
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        # JSONL
        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())

    print("RUN_DIR:", RUN_DIR)

    # ------------------------
    # Training loop (epochs)
    # ------------------------
    best_val = float("inf")
    opt.zero_grad(set_to_none=True)

    overall_t0 = time.time()
    ema_epoch_sec = None
    ema_alpha = 0.3

    n_classes = 3

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", leave=True):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        epoch_t0 = time.time()
        train_loss_sum = 0.0
        train_rec_sum = 0.0
        n_train_batches = 0

        # Patient coverage tracker (per epoch)
        patient_counter = Counter()

        pbar = tqdm(
            train_loader,
            desc=f"Train e{epoch:02d}",
            leave=False,
            mininterval=0.5,
            total=train_steps_per_epoch,
        )

        for i, (x, meta) in enumerate(pbar, start=1):
            if i > train_steps_per_epoch:
                # Stop after a fixed number of steps per epoch
                break

            # 1) Move to device
            x = x.to(device, non_blocking=True)  # [B, 100, 321]
            meta = meta.to(device, non_blocking=True)  # [B, 100, 5]

            # 2) Group labels {1,2,3} -> {0,1,2}
            y_group_raw = meta[:, 0, 1].long()  # [B]
            y_group = y_group_raw - 1
            valid_mask = (y_group >= 0) & (y_group < n_classes)

            if epoch == 1 and i == 1:
                print("Unique y_group first batch:", torch.unique(y_group).tolist())

            # Patient ID for SupCon: meta[:, 0, 0]
            patient_id = meta[:, 0, 0].long()  # [B]
            # Update coverage counter
            patient_counter.update(patient_id.cpu().tolist())

            # 3) AMP context
            if use_amp and device.type == "cuda":
                amp_ctx = autocast(device_type="cuda", dtype=amp_dtype)
            elif use_amp and device.type == "cpu":
                amp_ctx = torch.autocast("cpu")
            else:
                amp_ctx = torch.enable_grad()

            with amp_ctx:
                # 4) Forward student + teacher
                out_s = model(x, return_all=True)
                with torch.no_grad():
                    out_t = teacher.teacher(x, return_all=True)

                # 5) Individual losses
                #   - reconstruction
                L_rec = loss_rec(out_s["recon"], x)

                #   - group classification
                if valid_mask.any():
                    L_group = loss_group(out_s["logits"][valid_mask], y_group[valid_mask])
                else:
                    L_group = torch.tensor(0.0, device=device)

                #   - supervised contrastive on patient ID
                L_supcon = supcon_loss(out_s["proj"], patient_id)

                #   - consistency student-teacher on logits
                L_cons = consistency_loss(out_s["logits"], out_t["logits"])

                # 6) Total loss
                loss_step = (
                    weights["rec"] * L_rec
                    + weights["group"] * L_group
                    + weights["supcon"] * L_supcon
                    + weights["cons"] * L_cons
                )
                loss_for_backward = loss_step / accum_steps

            # 7) Backward + optimization (with gradient accumulation)
            scaler.scale(loss_for_backward).backward()

            if (i % accum_steps) == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                teacher.update(model)

            global_step += 1
            train_loss_sum += loss_step.item()
            train_rec_sum += L_rec.item()
            n_train_batches += 1

            pbar.set_postfix({"loss": f"{loss_step.item():.4f}"})

        train_loss = train_loss_sum / max(1, n_train_batches)
        train_rec = train_rec_sum / max(1, n_train_batches)

        # Patient coverage summary for this epoch
        if patient_counter:
            counts = np.array(list(patient_counter.values()))
            print(
                f"[epoch {epoch:02d}] patient coverage: "
                f"n_patients_seen={len(patient_counter)} | "
                f"min_batches={counts.min()} | max_batches={counts.max()} | "
                f"mean_batches={counts.mean():.2f}"
            )

        # --- Validation (sequential loader) ---
        model.eval()
        val_rec_sum = 0.0
        val_ce_sum = 0.0
        val_acc_sum = 0.0
        n_val_samples = 0

        with torch.no_grad():
            vbar = tqdm(
                val_loader,
                desc=f"Val   e{epoch:02d}",
                leave=False,
                mininterval=0.5,
            )
            for xv, meta_v in vbar:
                xv = xv.to(device, non_blocking=True)
                meta_v = meta_v.to(device, non_blocking=True)

                yv_raw = meta_v[:, 0, 1].long()
                yv = yv_raw - 1
                valid_mask = (yv >= 0) & (yv < n_classes)

                if use_amp and device.type == "cuda":
                    with autocast(device_type="cuda", dtype=amp_dtype):
                        out_v = model(xv, return_all=True)
                else:
                    out_v = model(xv, return_all=True)

                recon_v = out_v["recon"]
                logits_v = out_v["logits"]

                l_rec = loss_rec(recon_v, xv)
                if valid_mask.any():
                    l_ce = loss_group(logits_v[valid_mask], yv[valid_mask])
                else:
                    l_ce = torch.tensor(0.0, device=device)

                preds = logits_v.argmax(dim=-1)
                if valid_mask.any():
                    acc = (preds[valid_mask] == yv[valid_mask]).float().mean()
                else:
                    acc = torch.tensor(0.0, device=device)

                bs = xv.size(0)
                val_rec_sum += l_rec.item() * bs
                val_ce_sum += l_ce.item() * bs
                val_acc_sum += acc.item() * bs
                n_val_samples += bs

                vbar.set_postfix(
                    {
                        "rec": f"{l_rec.item():.4f}",
                        "ce": f"{l_ce.item():.4f}",
                        "acc": f"{acc.item():.3f}",
                    }
                )

        val_loss_rec = val_rec_sum / max(1, n_val_samples)
        val_loss_ce = val_ce_sum / max(1, n_val_samples)
        val_acc = val_acc_sum / max(1, n_val_samples)

        # Epoch timing, ETA
        epoch_secs = time.time() - epoch_t0
        ema_epoch_sec = (
            epoch_secs
            if ema_epoch_sec is None
            else (ema_alpha * epoch_secs + (1 - ema_alpha) * ema_epoch_sec)
        )
        eta_secs = ema_epoch_sec * (num_epochs - epoch)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | train_rec={train_rec:.6f} | "
            f"val_rec={val_loss_rec:.6f} | val_ce={val_loss_ce:.6f} | val_acc={val_acc:.4f} | "
            f"time/epoch={fmt_hms(epoch_secs)} | ETA total={fmt_hms(eta_secs)}"
        )

        # Log to files (val_loss = val_rec)
        log_epoch_row(
            epoch,
            global_step,
            train_loss,
            train_rec,
            val_loss_rec,
            val_loss_ce,
            val_acc,
            epoch_secs,
        )

        # Scheduler
        scheduler.step()

        # Checkpoints (best + last per run + last global)
        if val_loss_rec < best_val:
            best_val = val_loss_rec
            torch.save(model.state_dict(), RUN_DIR / "semisupae_subject_best.pt")
            print(
                f"[ckpt] new best → {RUN_DIR/'semisupae_subject_best.pt'} "
                f"(val_rec={best_val:.6f})"
            )

        save_ckpt(global_step, RUN_DIR / "semisupae_subject_last.pt")
        save_ckpt(global_step, CKPT_GLOBAL)

    # End of training
    total_secs = time.time() - overall_t0
    torch.save(model.state_dict(), RUN_DIR / "semisupae_subject_last_only_model.pt")
    print(
        f"[done] best val={best_val:.6f} | total={fmt_hms(total_secs)} | folder: {RUN_DIR}"
    )


if __name__ == "__main__":
    main()
