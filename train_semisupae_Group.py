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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


# (Ampere+): permitir TF32 para GEMMs
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# Wrappers usados en el loop
def loss_rec(out_recon, x):
    return F.mse_loss(out_recon, x)


def loss_group(logits, y):
    return F.cross_entropy(logits, y)

def loss_adv(nuis_logits, nuis):
    return F.cross_entropy(nuis_logits, nuis)

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
    micro_batch = 4096         # ajusta si OOM (por ejemplo 2048, 1024, etc.)
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
        GaitBatchIterable(train_path, batch_size=micro_batch, return_meta=True),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
    )

    val_loader = DataLoader(
        GaitBatchIterable(val_path, batch_size=micro_batch, return_meta=True),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
    )

    test_loader = DataLoader(
        GaitBatchIterable(test_path, batch_size=micro_batch, return_meta=True),
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch,
    )

    # Comprobación rápida del Zarr y DataLoader (solo una vez)
    p = Path(train_path)
    root = _zarr.open_group(str(p), mode="r")
    print("Arrays en train Zarr:", list(root.array_keys()))
    print("shape data train:", root["data"].shape)  # (N,100,326) esperado

    x0 = next(iter(train_loader))
    if isinstance(x0, (list, tuple)) and len(x0) == 2:
        xb0, meta0 = x0
        print("Primer batch x shape:", xb0.shape)
        print("Primer batch meta shape:", meta0.shape)
    else:
        print("Primer batch shape:", x0.shape)

    # Comprobación group labels en train_loader
    def inspect_group_labels(loader, max_batches=5):
        all_groups = []

        for b, (xb, metab) in enumerate(loader, start=1):
            g = metab[:, 0, 1]  # columna que asumimos es 'group'
            all_groups.append(g.flatten())

            print(f"\n[Batch {b}]")
            print("  min label:", g.min().item())
            print("  max label:", g.max().item())
            uniq = torch.unique(g)
            # en caso de que haya muchos valores, mostramos solo los primeros
            print("  unique labels (hasta 50):", uniq[:50].tolist())

            if b >= max_batches:
                break

        all_groups = torch.cat(all_groups)
        print("\n=== RESUMEN GLOBAL (primeros", max_batches, "batches) ===")
        print("Global min:", all_groups.min().item())
        print("Global max:", all_groups.max().item())
        uniq_global = torch.unique(all_groups)
        print("Unique global labels (hasta 100):", uniq_global[:100].tolist())
        print("Cantidad total de muestras inspeccionadas:", all_groups.numel())

    #print("\n>>> INSPECCIONANDO ETIQUETAS DE GRUPO EN TRAIN_LOADER...")
    #inspect_group_labels(train_loader, max_batches=3)
    #print("\n>>> FIN DE INSPECCIÓN. Ajusta según estos resultados antes de entrenar.")
    #return  # <- SALIMOS AQUÍ PARA NO LLEGAR AL LOOP DE ENTRENAMIENTO    

    # ------------------------
    # Modelo, EMA teacher y optimizador
    # ------------------------
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
    min_lr = 1e-5
    scheduler = CosineAnnealingLR(opt, T_max=num_epochs, eta_min=min_lr)

    scaler = GradScaler()

    weights = dict(rec=1, group=0.5, supcon=1.0, cons=0.1, adv=0.0, pseudo_w=0.0)

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

    RUN_NAME = "semisup_3groups"  # cambia el sufijo según config
    RUN_DIR = RUN_ROOT / f"{RUN_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    CSV_PATH = RUN_DIR / "metrics.csv"
    JSONL_PATH = RUN_DIR / "metrics.jsonl"
    FIELDS = ["epoch", "global_step",
          "train_loss", "train_rec", "val_loss", "val_ce", "val_acc",
          "seconds"]

    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log_epoch_row(epoch, global_step,
                  train_loss, train_rec, val_loss, val_ce, val_acc,
                  seconds):
        row = dict(
        epoch=int(epoch),
        global_step=int(global_step),
        train_loss=float(train_loss),
        train_rec=float(train_rec), 
        val_loss=float(val_loss),   # aquí guardamos val_rec
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
        train_rec_sum = 0.0
        n_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Train e{epoch:02d}", leave=False, mininterval=0.5)
        for i, (x, meta) in enumerate(pbar, start=1):
            # 1) Mover a GPU
            x    = x.to(device, non_blocking=True)       # [B, 100, 321]
            meta = meta.to(device, non_blocking=True)    # [B, 100, 5]

            # 2) Etiquetas de grupo (0,1,2) desde meta[:, 0, 1]
            y_group_raw = meta[:, 0, 1].long()               # [B]
            y_group = y_group_raw.long()
            y_group = y_group - 1
            n_classes = 3 
            valid_mask = (y_group >= 0) & (y_group < n_classes)

            if epoch == 1 and i == 1:
                print("Valores únicos y_group primer batch:", torch.unique(y_group).tolist())

            # (opcional) ID paciente para SupCon: meta[:, 0, 0]
            patient_id = meta[:, 0, 0].long()            # [B]

            # 3) AMP context
            with (
                autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp and device.type == "cuda"
                else torch.autocast("cpu")
                if use_amp and device.type == "cpu"
                else torch.enable_grad()
            ):
                # 4) Forward student y teacher
                out_s = model(x, return_all=True)
                with torch.no_grad():
                    out_t = teacher.teacher(x, return_all=True)

                # 5) Pérdidas individuales
                #   - recon (igual que antes, MSE)
                L_rec = loss_rec(out_s["recon"], x)

                #   - clasificación de grupo (NUEVA, usa logits vs y_group)
                if valid_mask.any():
                    L_group = loss_group(out_s["logits"][valid_mask], y_group[valid_mask])
                else:
                    L_group = torch.tensor(0.0, device=device)


                #   - SupCon (opcional, si quieres usarlo)
                L_supcon = supcon_loss(out_s["proj"], patient_id)


                #   - consistencia student-teacher en logits
                L_cons   = consistency_loss(out_s["logits"], out_t["logits"])

                # 6) Pérdida total (usando tu diccionario weights)
                loss_step = (
                    weights["rec"]    * L_rec +
                    weights["group"]  * L_group  
                    + weights["supcon"] * L_supcon 
                    + weights["cons"]   * L_cons
                )
                loss_for_backward = loss_step / accum_steps

            # 7) Backward y optimización
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
        train_rec  = train_rec_sum  / max(1, n_train_batches)

        # --- Validación ---
        model.eval()
        val_rec_sum = 0.0
        val_ce_sum  = 0.0
        val_acc_sum = 0.0
        n_val_samples = 0

        n_classes = 3

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Val   e{epoch:02d}", leave=False, mininterval=0.5)
            for xv, meta_v in vbar:
                # mover a device
                xv     = xv.to(device, non_blocking=True)
                meta_v = meta_v.to(device, non_blocking=True)

                # etiquetas {1,2,3} -> {0,1,2}
                yv_raw = meta_v[:, 0, 1].long()
                yv     = yv_raw - 1
                valid_mask = (yv >= 0) & (yv < n_classes)

                # forward
                if use_amp and device.type == "cuda":
                    with autocast(device_type="cuda", dtype=amp_dtype):
                        out_v = model(xv, return_all=True)
                else:
                    out_v = model(xv, return_all=True)

                recon_v  = out_v["recon"]
                logits_v = out_v["logits"]

                # pérdidas
                l_rec = loss_rec(recon_v, xv)  # MSE
                if valid_mask.any():
                    l_ce = loss_group(logits_v[valid_mask], yv[valid_mask])
                else:
                    l_ce = torch.tensor(0.0, device=device)

                # accuracy
                preds = logits_v.argmax(dim=-1)
                if valid_mask.any():
                    acc = (preds[valid_mask] == yv[valid_mask]).float().mean()
                else:
                    acc = torch.tensor(0.0, device=device)

                bs = xv.size(0)
                val_rec_sum += l_rec.item() * bs
                val_ce_sum  += l_ce.item()  * bs
                val_acc_sum += acc.item()   * bs
                n_val_samples += bs

                vbar.set_postfix({
                    "rec": f"{l_rec.item():.4f}",
                    "ce":  f"{l_ce.item():.4f}",
                    "acc": f"{acc.item():.3f}",
                })

        # medias por muestra
        val_loss_rec = val_rec_sum / max(1, n_val_samples)
        val_loss_ce  = val_ce_sum  / max(1, n_val_samples)
        val_acc      = val_acc_sum / max(1, n_val_samples)

        # tiempos y ETA total
        epoch_secs = time.time() - epoch_t0
        ema_epoch_sec = (
            epoch_secs if ema_epoch_sec is None else (ema_alpha * epoch_secs + (1 - ema_alpha) * ema_epoch_sec)
        )
        eta_secs = ema_epoch_sec * (num_epochs - epoch)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} |  train_rec={train_rec:.6f} |"
            f"val_rec={val_loss_rec:.6f} | val_ce={val_loss_ce:.6f} | val_acc={val_acc:.4f} | "
            f"time/epoch={fmt_hms(epoch_secs)} | ETA total={fmt_hms(eta_secs)}"
        )

        # logging a archivos (mantenemos val_loss como reconstrucción)
        log_epoch_row(
        epoch,
        global_step,
        train_loss,
        train_rec,
        val_loss_rec,   # va a la columna val_loss
        val_loss_ce,
        val_acc,
        epoch_secs,
        )

        # scheduler
        scheduler.step()

        # checkpoints (best + last por run + last global para reanudar)
        if val_loss_rec < best_val:
            best_val = val_loss_rec
            torch.save(model.state_dict(), RUN_DIR / "semisupae_best.pt")
            print(f"[ckpt] mejor → {RUN_DIR/'semisupae_best.pt'} (val_rec={best_val:.6f})")

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
