#!/usr/bin/env python
"""
Entrenamiento del SemiSupAE para la base de datos de marcha.

Ejecutar desde terminal:

    conda activate gait-stability
    python train_semisupae.py

Para correr en background:

    tmux new -s GaitAE
    cd /mnt/storage/dmartinez/Gait-Stability
    conda activate gait-stability
    python train_semisupae.py > train_semisupae_resume.log 2>&1
"""

# === Config & seeds ===
import os
import random
import time
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

# Semillas
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Performance (si no requieres reproducibilidad estricta)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# (Ampere+): permitir TF32 para GEMMs
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# === Pérdidas básicas ===
def consistency_loss(student_logits, teacher_logits):
    ps = F.softmax(student_logits, dim=-1)
    pt = F.softmax(teacher_logits.detach(), dim=-1)
    return F.mse_loss(ps, pt)


def supcon_loss(emb, labels, temperature=0.07):
    emb = F.normalize(emb, dim=-1)
    sim = torch.matmul(emb, emb.t()) / temperature
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.t()).float().to(sim.device)

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=sim.device)
    mask = mask * logits_mask

    log_prob = sim - torch.logsumexp(sim + torch.log(logits_mask + 1e-12), dim=1, keepdim=True)
    mean_log_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    return -mean_log_pos.mean()


# Wrappers usados en el loop
def loss_rec(out_recon, x):
    return F.mse_loss(out_recon, x)


def loss_group(logits, y):
    return F.cross_entropy(logits, y)


def loss_supcon(proj, pid, temperature=0.07):
    return supcon_loss(proj, pid, temperature)


def loss_cons(student_logits, teacher_logits):
    return consistency_loss(student_logits, teacher_logits)


def loss_adv(nuis_logits, nuis):
    return F.cross_entropy(nuis_logits, nuis)


# === DataLoader para Zarr ===
class GaitBatchIterable(IterableDataset):
    """
    Lee un Zarr con clave 'data' de forma (N, 100, 326)
    y devuelve solo las 321 features: x ~ (B, 100, 321).
    """

    def __init__(self, store_path, batch_size, return_meta=False):
        self.store_path = str(Path(store_path))
        self.bs = int(batch_size)
        self.return_meta = bool(return_meta)

    def __iter__(self):
        import zarr
        from torch.utils.data import get_worker_info

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            import numcodecs.blosc as nblosc

            nblosc.set_nthreads(1)  # 1 hilo por worker
        except Exception:
            pass

        p = Path(self.store_path)
        if not p.exists():
            raise FileNotFoundError(f"Zarr no encontrado: {p}")

        root = (
            zarr.open_consolidated(str(p), mode="r")
            if (p / ".zmetadata").exists()
            else zarr.open_group(str(p), mode="r")
        )
        key = "data" if "data" in root else root.array_keys()[0]
        z = root[key]  # (N, 100, 326)
        n = len(z)

        wi = get_worker_info()
        rng = np.random.default_rng(wi.id if wi else None)

        starts = np.arange(0, n, self.bs, dtype=np.int64)
        rng.shuffle(starts)

        if wi:
            wid, nw = wi.id, wi.num_workers
            total = len(starts)
            per, rem = total // nw, total % nw
            beg = wid * per + min(wid, rem)
            end = beg + per + (1 if wid < rem else 0)
            starts = starts[beg:end]

        for s in starts:
            lo, hi = int(s), int(min(s + self.bs, n))
            # leer solo las 321 primeras columnas
            data = z[lo:hi, :, :321].astype("float32", copy=True)  # (B,100,321)
            feat = torch.from_numpy(data)  # (B,100,321)
            yield feat


# === Utilidades varias ===
def get_recon(out):
    # dict (semisup), tuple (AE) o tensor
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


# === Modelo y entrenamiento ===
def main():
    # ------------------------
    # Hiperparámetros
    # ------------------------
    micro_batch = 4096          # ajusta si OOM (por ejemplo 2048, 1024, etc.)
    accum_steps = 1             # con batch grande puedes dejar 1
    num_workers = 8
    prefetch = 6

    num_epochs = 20
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16   # en Ampere va bien; puedes cambiar a float16 si lo prefieres

    print(f"micro_batch={micro_batch} | accum_steps={accum_steps} | "
          f"workers={num_workers} | prefetch={prefetch}")

    # ------------------------
    # Rutas de datos
    # ------------------------
    train_path = str(Path("train_cycles.zarr").resolve())
    val_path = str(Path("val_cycles.zarr").resolve())
    test_path = str(Path("test_cycles.zarr").resolve())  # opcional

    # DataLoaders
    train_loader = DataLoader(
        GaitBatchIterable(train_path, batch_size=micro_batch, return_meta=False),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
    )

    val_loader = DataLoader(
        GaitBatchIterable(val_path, batch_size=micro_batch, return_meta=False),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
    )

    test_loader = DataLoader(
        GaitBatchIterable(test_path, batch_size=micro_batch, return_meta=False),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
    )

    # Comprobación rápida del Zarr y DataLoader (solo una vez)
    import zarr as _zarr

    p = Path(train_path)
    root = _zarr.open_group(str(p), mode="r")
    print("Arrays en train Zarr:", list(root.array_keys()))
    print("shape data train:", root["data"].shape)  # (N,100,326) esperado

    x0 = next(iter(train_loader))
    print("Primer batch shape:", x0.shape)  # (B,100,321)

    # ------------------------
    # Modelo, EMA teacher y optimizador
    # ------------------------
    from AE_pipeline_pytorch import SemiSupAE, EMATeacher

    # OJO: ahora hay 3 grupos (joven, middle, older)
    model = SemiSupAE(
        steps=100,
        in_dim=321,
        latent=128,
        n_group=3,
        n_nuisance=None,
    ).to(device)

    teacher = EMATeacher(model, ema=0.995)  # copia congelada con EMA

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    warmup_steps = 1500
    min_lr = 1e-5
    scheduler = CosineAnnealingLR(opt, T_max=(num_epochs * 1500), eta_min=min_lr)

    scaler = GradScaler()

    weights = dict(rec=1.0, group=1.0, supcon=0.5, cons=0.5, adv=0.1, pseudo_w=0.3)

    # (Opcional) PyTorch 2.x: compilar (descomenta si quieres probar)
    # model = torch.compile(model)
    # teacher.teacher = torch.compile(teacher.teacher)

    # ------------------------
    # Checkpointing
    # -----------------------

    CKPT_DIR = Path("checkpoints_3groups")
    CKPT_DIR.mkdir(exist_ok=True)
    CKPT_GLOBAL = CKPT_DIR / "semisupae_last.pt"

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
        print(f"[ckpt] guardado en {path} (step={step})")

    def load_ckpt(path):
        if not path.exists():
            print("[ckpt] no encontrado, inicio desde cero")
            return 0
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        teacher.teacher.load_state_dict(ckpt["teacher"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        print(f"[ckpt] cargado desde {path} (step={ckpt.get('step', 0)})")
        return ckpt.get("step", 0)

    global_step = load_ckpt(CKPT_GLOBAL)

    # ------------------------
    # Setup logging por época (CSV + JSONL)
    # ------------------------
    RUN_ROOT = Path("runs")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    RUN_NAME = "semisup_3groups_bs4096"  # cambia el sufijo según config
    RUN_DIR = RUN_ROOT / f"{RUN_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    CSV_PATH = RUN_DIR / "metrics.csv"
    JSONL_PATH = RUN_DIR / "metrics.jsonl"
    FIELDS = ["epoch", "global_step", "train_loss", "val_loss", "seconds"]

    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log_epoch_row(epoch, global_step, train_loss, val_loss, seconds):
        row = dict(
            epoch=int(epoch),
            global_step=int(global_step),
            train_loss=float(train_loss),
            val_loss=float(val_loss),
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
    # Loop de entrenamiento por épocas
    # ------------------------
    best_val = float("inf")
    opt.zero_grad(set_to_none=True)

    overall_t0 = time.time()
    ema_epoch_sec = None
    ema_alpha = 0.3

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", leave=True):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        epoch_t0 = time.time()
        train_loss_sum = 0.0
        n_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Train e{epoch:02d}", leave=False, mininterval=0.5)
        for i, x in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)

            ctx = (
                autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp and device.type == "cuda"
                else torch.autocast("cpu")
                if use_amp and device.type == "cpu"
                else torch.cuda.amp.autocast(enabled=False)
            )

            # El contexto "ctx" solo funciona en CUDA; en CPU puedes simplemente no usar amp
            if device.type != "cuda":
                ctx = torch.no_grad() if False else torch.enable_grad()

            with (
                autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp and device.type == "cuda"
                else torch.autocast("cpu")
                if use_amp and device.type == "cpu"
                else torch.enable_grad()
            ):
                out_s = model(x, return_all=True)
                with torch.no_grad():
                    out_t = teacher.teacher(x, return_all=True)

                loss_step = (
                    weights["rec"] * loss_rec(out_s["recon"], x)
                    + weights["cons"] * loss_cons(out_s["logits"], out_t["logits"])
                )
                loss_for_backward = loss_step / accum_steps

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
            n_train_batches += 1

            pbar.set_postfix({"loss": f"{loss_step.item():.4f}"})

        train_loss = train_loss_sum / max(1, n_train_batches)

        # --- Validación ---
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Val   e{epoch:02d}", leave=False, mininterval=0.5)
            for vb in vbar:
                xv = (
                    vb.to(device, non_blocking=True)
                    if torch.is_tensor(vb)
                    else vb[0].to(device, non_blocking=True)
                )
                with (
                    autocast(device_type="cuda", dtype=amp_dtype)
                    if use_amp and device.type == "cuda"
                    else torch.autocast("cpu")
                    if use_amp and device.type == "cpu"
                    else torch.enable_grad()
                ):
                    out_v = model(xv)
                recon_v = get_recon(out_v)
                l = loss_rec(recon_v, xv).item()
                val_loss_sum += l
                n_val_batches += 1
                vbar.set_postfix({"loss": f"{l:.4f}"})

        val_loss = val_loss_sum / max(1, n_val_batches)

        # tiempos y ETA total
        epoch_secs = time.time() - epoch_t0
        ema_epoch_sec = (
            epoch_secs if ema_epoch_sec is None else (ema_alpha * epoch_secs + (1 - ema_alpha) * ema_epoch_sec)
        )
        eta_secs = ema_epoch_sec * (num_epochs - epoch)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"time/epoch={fmt_hms(epoch_secs)} | ETA total={fmt_hms(eta_secs)}"
        )

        # logging a archivos
        log_epoch_row(epoch, global_step, train_loss, val_loss, epoch_secs)

        # scheduler
        scheduler.step()

        # checkpoints (best + last por run + last global para reanudar)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), RUN_DIR / "semisupae_best.pt")
            print(f"[ckpt] mejor → {RUN_DIR/'semisupae_best.pt'} (val={best_val:.6f})")

        save_ckpt(global_step, RUN_DIR / "semisupae_last.pt")
        save_ckpt(global_step, CKPT_GLOBAL)

    # fin entrenamiento
    total_secs = time.time() - overall_t0
    torch.save(model.state_dict(), RUN_DIR / "semisupae_last_only_model.pt")
    print(
        f"[fin] mejor val={best_val:.6f} | total={fmt_hms(total_secs)} | carpeta: {RUN_DIR}"
    )


if __name__ == "__main__":
    main()
