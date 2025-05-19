import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np

from Data_loader  import (
    base_folders,
    ensure_dir,
    list_patient_ids,
    iter_trial_paths,
    load_patient_data,
    summarize_file
)
from SpatioTemporal_calculation import process_spatiotemporal_for_patient
from summary_utils import save_trial_summary, ensure_dir
from downsample import downsample_df
from scipy.signal import decimate
from gait_events import gait_events_HC_JA

# ─── Downsample ─────────────────────────────────────────────────
def process_group_downsample(group_code, downsample_rate, cols=None, verbose=False):
    """
    For each trial CSV in group_code:
     1) Load the raw CSV via pandas
     2) Downsample via downsample_df()
     3) Save to <base>/<group_code>/<patiend_id>/downsampled/<trial>.csv
    """
    base = base_folders[group_code]
    
    for pid in tqdm(list_patient_ids(base),
                    desc=f"Downsampling {group_code}",
                    unit="patient"):
        patient_src = os.path.join(base, pid)
        patient_out = os.path.join(patient_src, "downsampled")
        ensure_dir(patient_out)

        for day, block, trial, path in iter_trial_paths(patient_src, pid, group_code):
            if not os.path.isfile(path):
                if verbose:
                    print(f"  [WARN] Missing file: {path}")
                continue

            try:
                # 1) read raw trial
                df_raw = pd.read_csv(path)
                # 2) downsample all columns (or pass cols=[…] to limit)
                df_ds  = downsample_df(df_raw, rate=downsample_rate, cols=cols)
                # 3) write out
                fname = f"{pid}_{group_code}_{day}_{block}_{trial}.csv"
                out_path = os.path.join(patient_out, fname)
                df_ds.to_csv(out_path, index=False)
                if verbose:
                    print(f"  [OK] {fname}")
            except Exception as e:
                print(f"  [ERROR] {path}: {e}")

# ─── Segment gait cycle base in gait events ─────────────────────────────────────────────────
def segment_downsamp(df,
                   signal_col='Ankle Dorsiflexion RT (deg)',
                   min_length=20,
                   downsample_factor=4):
    hs_R, _, _, _ = gait_events_HC_JA(df)
    if len(hs_R) < 2:
        return []
    series = df[signal_col].interpolate().values
    cycles = []
    hs = sorted(int(i) for i in hs_R)
    for start, end in zip(hs, hs[1:]):
        cycle = series[start:end]
        if len(cycle) >= min_length:
            if len(cycle) > 27:
                cycle = (cycle - np.mean(cycle)) / np.std(cycle)
                cycle = decimate(cycle, downsample_factor, zero_phase=True)
                cycles.append(cycle)
    return cycles

def segment_patient_cycles(patient_id, group_code, source="raw", verbose=False):
    
    """
    1) Load each trial for one patient (raw or downsampled)
    2) Segment into cycles in memory
    3) Compute DTW matrix and summary
    4) Save one CSV with mean/std distances for this patient
    """
    base       = base_folders[group_code]
    data_folder= base if source=="raw" else os.path.join(base, "downsampled")
    segments    = {}

    for day, block, trial, path in iter_trial_paths(data_folder, patient_id, group_code):
        if not os.path.isfile(path):
            if verbose: print(f"[WARN] missing file: {path}")
            continue

        df = pd.read_csv(path)
        cycles = segment_cycles(df)
        if not cycles:
            if verbose: print(f"[WARN] no cycles found in {os.path.basename(path)}")
            continue

        key = f"{day}_{block}_{trial}"
        segments[key] = cycles
        if verbose:
            print(f"[INFO] {patient_id} {key}: {len(cycles)} cycles")

    return segments

def get_segments_for_group(group_code,
                           source='raw',
                           verbose=False):
    """
    Run segmentation for every patient in a group, returning a nested dict:
        { patient_id: { trial_name: [cycle_dfs] } }
    """
    all_segs = {}
    base = base_folders[group_code]

    for pid in tqdm(list_patient_ids(base),
                    desc=f"Segmenting ({source}) {group_code}",
                    unit="patient"):
        if verbose:
            print(f"\n[INFO] Patient {pid}")
        segs = get_segments_for_patient(pid, group_code, source, verbose)
        if segs:
            all_segs[pid] = segs

    return all_segs

# ─── Raw DataBase Analysis ─────────────────────────────────────────────────

def process_one_patient(patient_id, group_code, verbose=False):
    """
    In case I dont want to process all patients in a group, I can run this function
    to process a single patient.
    It will:
    1) Iterate all trial CSVs for one patient
    2) Summarize each file
    3) Save per-trial CSV summaries
    """
    base          = base_folders[group_code]
    patient_folder= os.path.join(base, patient_id)
    out_folder    = os.path.join(".", "EDA", group_code, patient_id)
    ensure_dir(out_folder)

    for day, block, trial, path in iter_trial_paths(patient_folder, patient_id, group_code):
        if not os.path.exists(path):
            if verbose:
                print(f"[WARN] missing: {path}")
            continue

        try:
            # 1) read & describe
            summary = summarize_file(path)
            # 2) save and free memory
            save_trial_summary(summary, out_folder, patient_id, day, block, trial)
            del summary
        except Exception as e:
            print(f"[ERROR] processing {path}: {e}")


def process_group(group_code, verbose=False):
    """
    1) Iterate every trial CSV for each patient in group_code
    2) Compute per-trial summary via summarize_file()
    3) Save summary CSV via save_trial_summary()
    """
    # ensure the group folder exists under EDA/
    ensure_dir(os.path.join(".", "EDA", group_code))

    base = base_folders[group_code]
    for pid in tqdm(list_patient_ids(base),
                    desc=f"Group {group_code}",
                    unit="patient"):
        if verbose:
            print(f"\n[INFO] Patient {pid} ({group_code})")

        patient_folder = os.path.join(base, pid)
        out_folder     = os.path.join(".", "EDA", group_code, pid)
        ensure_dir(out_folder)

        for day, block, trial, path in iter_trial_paths(patient_folder, pid, group_code):
            if not os.path.exists(path):
                if verbose:
                    print(f"[WARN] missing: {path}")
                continue

            try:
                # 1) read & summarize
                summary = summarize_file(path)
                # 2) save and free memory
                save_trial_summary(summary, out_folder, pid, day, block, trial)
                del summary
            except Exception as e:
                print(f"[ERROR] processing {path}: {e}")



# ─── SpatioTemporal Variables Calculation ─────────────────────────────────────────────────
def process_patient_spatiotemporal(patient_id,
                                   group_code,
                                   sampling_rate,
                                   verbose=False):
    """
    Compute spatiotemporal (mean & std) for a single patient.
    Saves:
      • EDA/SpTVariables/Mean/<group_code>/<patient_id>/<patient_id>_spatiotemporal_mean.csv
      • EDA/SpTVariables/STD/ <group_code>/<patient_id>/<patient_id>_spatiotemporal_std.csv
    """
    # 1) Load this patient’s full DataFrame
    base = base_folders[group_code]
    df   = load_patient_data(os.path.join(base, patient_id),
                             patient_id, group_code,
                             verbose=verbose)
    if df.empty:
        if verbose:
            print(f"[WARN] No data for {patient_id} in {group_code}")
        return

    # 2) Create a temporary folder to let the existing function write both files
    temp_out = os.path.join("EDA", "SpTVariables", "temp", group_code, patient_id)
    ensure_dir(temp_out)

    # 3) Call the spatiotemporal routine (it will write:
    #      <temp_out>/<patient_id>_spatiotemporal_mean.csv
    #      <temp_out>/<patient_id>_spatiotemporal_std.csv
    process_spatiotemporal_for_patient(
        patient_df    = df,
        patient_id    = patient_id,
        output_folder = temp_out,
        sampling_rate = sampling_rate,
        verbose       = verbose
    )

    # 4) Move each file into its “Mean” or “STD” permanent folder
    for kind in ("mean", "std"):
        src = os.path.join(
            temp_out,
            f"{patient_id}_spatiotemporal_{kind}.csv"
        )
        if not os.path.exists(src):
            continue
        dest_folder = os.path.join(
            "EDA", "SpTVariables",
            "Mean" if kind=="mean" else "STD",
            group_code,
            patient_id
        )
        ensure_dir(dest_folder)
        shutil.move(src, os.path.join(dest_folder, os.path.basename(src)))

    # 5) Clean up temp
    shutil.rmtree(os.path.join("EDA", "SpTVariables", "temp", group_code, patient_id),
                  ignore_errors=True)
    if verbose:
        print(f"[INFO] Patient {patient_id} spatiotemporal results saved.")


def process_group_spatiotemporal(group_code,
                                 sampling_rate,
                                 verbose=False):
    """
    Apply spatiotemporal extraction to every patient in `group_code`.
    """
    base = base_folders[group_code]
    ids  = list_patient_ids(base)

    for pid in tqdm(ids,
                    desc=f"SpTVariables {group_code}",
                    unit="patient"):
        if verbose:
            print(f"\n[INFO] Processing {pid}")
        process_patient_spatiotemporal(
            patient_id    = pid,
            group_code    = group_code,
            sampling_rate = sampling_rate,
            verbose       = verbose
        )

