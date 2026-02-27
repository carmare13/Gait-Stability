import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import random
import re
import glob 
from pathlib import Path
from sklearn.preprocessing import StandardScaler  # Z-score normalization
from sklearn.preprocessing import MinMaxScaler  # Min-Max normalization
from joblib import dump, load  # To save and load the scaler
from scipy.signal import decimate
from scipy.interpolate import interp1d

from Data_loader  import (
    base_folders,
    ensure_dir,
    list_patient_ids,
    iter_trial_paths,
    load_patient_data,
    load_and_clean_csv,
    summarize_file
)
from SpatioTemporal_calculation import process_spatiotemporal_for_patient
from summary_utils import save_trial_summary, ensure_dir
from downsample import downsample_df
from gait_events import gait_events_HC_JA
from segment_utils import segment_cycles_simple


# Folder where this .py lives (assumes the 3 trial characteristics CSVs are in the same folder). We use this to load trial characteristics for walking direction.
TRIAL_CHAR_DIR = os.path.dirname(os.path.abspath(__file__))

_TRIAL_CHAR_CACHE: dict[str, pd.DataFrame] = {}

# ─── Trim signal edges to avoid starting and ending noise ───────────────────────────────────────
def trim_signal_edges(df, fs, trim_seconds):
    """
    Trim the first and last trim_seconds from the signal.
    
    Args:
        df: pandas DataFrame containing the signal data.
        fs: sampling frequency in Hz.
        trim_seconds: number of seconds to trim from start and end.
        
    Returns:
        Trimmed pandas DataFrame.
    """
    n_trim = int(trim_seconds * fs)
    
    if len(df) <= 2 * n_trim:
        # Signal too short to trim, return empty DataFrame
        return pd.DataFrame(columns=df.columns)
    
    # Trim rows: skip first n_trim and last n_trim samples
    df_trimmed = df.iloc[n_trim:-n_trim].reset_index(drop=True)
    return df_trimmed

def process_patient_trim(pid, group_code, fs, trim_seconds, verbose=True):
    """
    Iterates through all trials of a patient (pid) in the group group_code,
    applies trim_signal_edges to each trial, saves the trimmed CSV, and deletes the original.

    Args:
        pid: patient identifier (string).
        group_code: group code (for example, 'groupA', 'groupB', etc.).
        fs: sampling frequency in Hz.
        trim_seconds: seconds to trim at the start and end of each trial.
        verbose: if True, prints progress/warning messages.
    """

    # Base folder for that group (according to your Data_loader).
    # We assume base_folders is a dict: base_folders[group_code] -> main path
    if group_code not in base_folders:
        raise ValueError(f"The group_code '{group_code}' is not in base_folders.")

    base = base_folders[group_code]
    patient_src = os.path.join(base, pid)
    patient_out = os.path.join(base, pid, "trimmed")

    # Create the output directory if it doesn't exist
    ensure_dir(patient_out)

    # Iterate over each trial: day, block, trial, and CSV file path
    for day, block, trial, path in iter_trial_paths(patient_src, pid, group_code):
        # If the file doesn't exist, warn and continue
        if not os.path.isfile(path):
            if verbose:
                print(f"  [WARN] Missing file: {path}")
            continue

        try:
            # 1) Load raw CSV
            df_raw = pd.read_csv(path)

            # 2) Apply trimming
            df_trimmed = trim_signal_edges(df_raw, fs, trim_seconds)

            # 3) If the trimmed DataFrame is empty, warn and skip
            if df_trimmed.empty:
                if verbose:
                    print(f"  [WARN] Trimmed data is empty for: {path}")
                continue

            # 4) Construct output filename and save trimmed CSV
            fname = f"{pid}_{group_code}_{day}_{block}_{trial}.csv"
            out_path = os.path.join(patient_out, fname)
            df_trimmed.to_csv(out_path, index=False)

            # 5) Remove the original file (optional)
            try:
                os.remove(path)
                if verbose:
                    print(f"  [OK] Saved: {fname} (original removed)")
            except Exception as e_rm:
                print(f"  [ERROR] Could not remove '{path}': {e_rm}")

        except Exception as e:
            print(f"  [ERROR] Processing '{path}': {e}")


def process_group_trim(group_code, fs, trim_seconds, verbose=False):
    """
    For each trial CSV in the group:
    1) Load the raw CSV via pandas
    2) Trim edges of the signal
    3) Save to <base>/<patient_id>/trimmed/<trial>.csv
    4) Erases the original file
    """
    base = base_folders[group_code]
    
    for pid in tqdm(list_patient_ids(base), desc=f"Trimming group {group_code}", unit="patient"):
        patient_src = os.path.join(base, pid)
        patient_out = os.path.join(base,pid, "trimmed")
        ensure_dir(patient_out)

        for day, block, trial, path in iter_trial_paths(patient_src, pid, group_code):
            if not os.path.isfile(path):
                if verbose:
                    print(f"  [WARN] Missing file: {path}")
                continue

            try:
                # Load raw trial data
                df_raw = pd.read_csv(path)
                
                # Trim the edges
                df_trimmed = trim_signal_edges(df_raw, fs, trim_seconds=trim_seconds)
                
                if df_trimmed.empty:
                    if verbose:
                        print(f"  [WARN] Trimmed empty data for file: {path}")
                    continue
                
                # Save trimmed file
                fname = f"{pid}_{group_code}_{day}_{block}_{trial}.csv"
                out_path = os.path.join(patient_out, fname)
                df_trimmed.to_csv(out_path, index=False)
                try:
                    # Remove the original file
                    os.remove(path)
                    if verbose:
                        print(f"  [OK] {fname} and original removed")
                except Exception as e:
                    print(f"  [ERROR] erasing {path}: {e_rm}")

            except Exception as e:
                    print(f"  [ERROR] processing {path}: {e}")

# ─── Downsample ─────────────────────────────────────────────────
def downsample_df(df, rate, cols=None, zero_phase=True):
    """
    Downsample a pandas DataFrame by an integer factor `rate`.

    Args:
        df      (pd.DataFrame): time-series indexed by time (or integer index).
        rate    (int): factor by which to decimate (keep 1 sample every `rate`).
        cols    (List[str], optional): which columns to downsample. If None, all.
        zero_phase (bool): if True, uses zero-phase filtering (via `filtfilt` 
                           internally in `decimate`), to avoid phase shift.

    Returns:
        pd.DataFrame: downsampled DataFrame, with the same index (kept) or reset.
    """
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=cols)

    to_ds = cols if cols is not None else df.columns.tolist()
        
    out = {c: decimate(df[c].values, rate, ftype='iir', zero_phase=zero_phase) for c in to_ds}
    
    # build new DataFrame; index will be compressed accordingly
    # if your index is numeric time and you want to keep it, you can do:
    if hasattr(df.index, 'values'):
        new_index = df.index.values[::rate][: len(out[to_ds[0]])]
    else:
        new_index = range(len(out[to_ds[0]]))
    
    result = pd.DataFrame(out, index=new_index)
    
    # if there are other columns you didn’t downsample, copy them via slicing
    if cols is not None and set(cols) != set(df.columns):
        extra_cols = df.columns.difference(cols)
        extras = df.loc[:, extra_cols].iloc[::rate].reset_index(drop=True)
        extras.index = result.index  
        result = pd.concat([result, extras], axis=1)
    
    return result

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
                   downsample_factor=4,
                   print_length=False
                   ):
    """
        Segment a downsampled DataFrame into gait cycles.
        Normalizes using z-score
        Optionally prints the length of each cycle.
    """
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
                if print_length:
                    print(f"Cycle length: {len(cycle)}")
                cycle = (cycle - np.mean(cycle)) / np.std(cycle)
                cycle = decimate(cycle, downsample_factor, zero_phase=True)
                cycles.append(cycle)
    return cycles

def segment_patient_cycles(patient_id, group_code, source="raw", verbose=False):
    
    """
    1) Load each trial for one patient (raw or downsampled)
    2) Segment into cycles in memory
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
        segs = segment_patient_cycles(pid, group_code, source, verbose)
        if segs:
            all_segs[pid] = segs

    return all_segs

def process_group_cycle_stats(group_code, verbose=False):
    """
    For each patient in the given group:
      1) Load each trial CSV from the trimmed folder
      2) Segment cycles with segment_cycles_simple(df, print_cycle_length=True)
      3) Collect the length (number of samples) of every cycle
      4) After processing all trials for a patient, print the min, max, and mean cycle length
    """
    base = base_folders[group_code]
    
    for patient_id in tqdm(list_patient_ids(base), desc=f"Processing group {group_code}", unit="patient"):
        patient_dir = os.path.join(base, patient_id, "trimmed")
        all_cycle_lengths = []
        
        for day, block, trial, path in iter_trial_paths(patient_dir, patient_id, group_code):
            if not os.path.isfile(path):
                if verbose:
                    print(f"  [WARN] Missing file: {path}")
                continue
            
            try:
                df = pd.read_csv(path)
                # Segment cycles and print each cycle's length to stdout
                cycles = segment_cycles_simple(df, print_cycle_length=False)
                # Append the length (number of samples) of each cycle
                all_cycle_lengths += [len(cycle) for cycle in cycles]
            except Exception as e:
                print(f"  [ERROR] Processing {path}: {e}")
        
        if all_cycle_lengths:
            minimum = min(all_cycle_lengths)
            maximum = max(all_cycle_lengths)
            average = sum(all_cycle_lengths) / len(all_cycle_lengths)
            print(f"Patient {patient_id}: min={minimum}, max={maximum}, mean={average:.2f} samples")
        else:
            print(f"Patient {patient_id}: no cycles detected\n")

def process_group_cycle_counts(group_code: str, verbose: bool = False, save_to_file: bool = False) -> dict[str, int]:
    """
    For each patient in the specified group:
      1) Iterate through each trial's CSV file in the "trimmed" folder.
      2) Segment cycles using segment_cycles_simple(df).
      3) Sum the number of cycles obtained across all trials.
      4) Print (and return) the total cycle count per patient.

    Args:
      group_code (str): Group code (e.g., "G01" or "G03").
      verbose (bool): If True, prints warnings for missing files.
      save_to_file (bool): If True, saves the results to a JSON file named
                            '[group_code]_cycle_counts.json' in the project directory.

    Returns:
      dict[str, int]: Maps each patient_id to the total number of detected cycles.
    """
    base = base_folders[group_code]
    patient_cycle_counts: dict[str, int] = {}

    for patient_id in tqdm(list_patient_ids(base), desc=f"Processing group {group_code}", unit="patient"):
        patient_dir = os.path.join(base, patient_id, "trimmed")
        total_cycles = 0

        for day, block, trial, path in iter_trial_paths(patient_dir, patient_id, group_code):
            if not os.path.isfile(path):
                if verbose:
                    print(f"  [WARN] Missing file: {path}")
                continue

            try:
                df = pd.read_csv(path)
                # Segment the cycles for this trial (we don't print lengths here)
                cycles = segment_cycles_simple(df, print_cycle_length=False)
                # Sum the number of cycles obtained in this trial
                total_cycles += len(cycles)
            except Exception as e:
                print(f"  [ERROR] Processing {path}: {e}")

        patient_cycle_counts[patient_id] = total_cycles
        print(f"Patient {patient_id}: {total_cycles} cycles detected")

    if save_to_file:
        output_filename = f"{group_code}_cycle_counts.json"
        try:
            with open(output_filename, 'w') as f:
                json.dump(patient_cycle_counts, f, indent=4)
            print(f"Cycle counts saved to {output_filename}")
        except Exception as e:
            print(f"[ERROR] Could not save to file {output_filename}: {e}")

    return patient_cycle_counts

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

# ─── Temporal normalization 0-100% gait cycle ─────────────────────────────────────────────────

def temporal_normalization_GC(df, target_length):
    """
    Given a DataFrame,
    segments it into cycles using segment_cycles_simple(), and temporally normalizes
    each cycle to have exactly 'target_length' samples using interp1d
    Returns a 3D tensor of shape (n_cycles, target_length, n_variables).
    
    Args:
        df (pd.DataFrame): DataFrame for a trial with 321 columns of kinematic signals.
        target_length (int): Fixed number of samples desired per cycle after temporal normalization.
        
    Returns:
        np.ndarray: Array of shape (n_cycles, target_length, n_variables).
                    If no cycles are detected, returns an empty array with shape (0, target_length, n_variables).
    """
    # 1) Segment into cycles
    cycles = segment_cycles_simple(df, print_cycle_length=False)
    if not cycles:
        # No cycles detected
        return np.empty((0, target_length, df.shape[1]), dtype=float)
    
    n_variables = df.shape[1]
    normalized_cycles = []

    # 2) For each cycle, interpolate to target_length
    for cycle_df in cycles:
        original_length = cycle_df.shape[0]
        # Create normalized index vectors [0, 1]
        original_idx = np.linspace(0, 1, original_length)
        target_idx = np.linspace(0, 1, target_length)
        
        # Matrix for the normalized cycle
        norm_cycle = np.zeros((target_length, n_variables), dtype=float)
        
        # Interpolate column by column
        for var_idx in range(n_variables):
            series = cycle_df.iloc[:, var_idx].values
            interp_func = interp1d(original_idx, series, kind='linear', fill_value="extrapolate")
            norm_cycle[:, var_idx] = interp_func(target_idx)
        
        normalized_cycles.append(norm_cycle)

    # 3) Stack all cycles into a 3D tensor
    tensor_3d = np.stack(normalized_cycles, axis=0)  # Shape: (n_cycles, target_length, n_variables)
    return tensor_3d

# ─── Characteristics normalization  ─────────────────────────────────────────────────
def set_scaler(
        training_patient_ids: list[str],
        target_length: int,
        num_kinematic_var: int,
        scaler_type: str = "standard",
        scaler_filename: str = "global_kinematic_scaler.pkl",
        checkpoint_every: int = 5,
        load_checkpoint: bool = False
    ):
    """
    Build and save a global scaler for kinematic data, with checkpointing support.

    Args:
        training_patient_ids: List of patient IDs to use for fitting.
        target_length: Number of timesteps to normalize each cycle to.
        num_kinematic_var: Number of kinematic variables (columns) in the dataset.
        scaler_type: "standard" or "minmax".
        scaler_filename: Name of file to save final scaler.
        checkpoint_every: Frequency (in patients) to save scaler checkpoint.
        load_checkpoint: Whether to resume from an existing checkpoint.
    """

    checkpoint_file = scaler_filename.replace(".pkl", "_checkpoint.pkl")

    # Select scaler
    scaler_type = scaler_type.lower()
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'.")

    # Optionally load checkpoint
    start_index = 0
    if load_checkpoint and os.path.exists(checkpoint_file):
        scaler = load(checkpoint_file)
        tqdm.write(f"[INFO] Loaded checkpoint from: {checkpoint_file}")
        # Optional: track how many patients were already processed if desired
        # For now, we assume the user adjusts training_patient_ids manually if needed

    # Loop over patients
    for idx, patient_id in enumerate(tqdm(training_patient_ids, desc="Patients", unit="pt")):
        found = False

        for group_code, base_folder in base_folders.items():
            if patient_id not in list_patient_ids(base_folder):
                continue

            patient_folder = os.path.join(base_folder, patient_id, "trimmed_sc")
            trials_list, _ = load_patient_data(patient_folder, patient_id, group_code, verbose=True)

            if not trials_list:
                tqdm.write(f"[INFO] No data for patient {patient_id} in {patient_folder}.")
                found = True
                break

            for trial_df in tqdm(trials_list, desc=f"Trials {patient_id}", unit="tr", leave=False):
                # Drop time column if present (case-insensitive)
                time_cols = [c for c in trial_df.columns if c.lower() == "time"]
                if time_cols:
                    trial_df = trial_df.drop(columns=time_cols)

                kinematic_df = trial_df.iloc[:, :num_kinematic_var]
                if kinematic_df.shape[1] != num_kinematic_var:
                    tqdm.write(f"[WARN] {patient_id}: got {kinematic_df.shape[1]} kinematic cols after dropping time. Skipping.")
                    continue
                normalized_cycles = temporal_normalization_GC(
                    kinematic_df, target_length=target_length
                )

                if normalized_cycles.size == 0:
                    tqdm.write(f"[WARN] No valid cycles for {patient_id} trial {trial_df['trial'].iloc[0]}.")
                    continue

                reshaped = normalized_cycles.reshape(-1, num_kinematic_var)
                scaler.partial_fit(reshaped)

            found = True
            break  # exit group loop once patient is processed

        if not found:
            tqdm.write(f"[ERROR] Patient {patient_id} not found in any group.")

        # Save checkpoint periodically
        if (idx + 1) % checkpoint_every == 0:
            dump(scaler, checkpoint_file)
            tqdm.write(f"[CHECKPOINT] Saved checkpoint at patient {patient_id}")

    # Final verification
    try:
        if isinstance(scaler, StandardScaler):
            _ = scaler.mean_
        elif isinstance(scaler, MinMaxScaler):
            _ = scaler.data_min_
    except AttributeError:
        raise ValueError("No data processed: scaler was not fitted.")

    # Save final scaler
    dump(scaler, scaler_filename)
    tqdm.write(f"[INFO] Final scaler saved to: {scaler_filename}")

    # Show some stats
    if isinstance(scaler, StandardScaler):
        tqdm.write(f"Means (first 5): {scaler.mean_[:5]}")
        tqdm.write(f"Scales (first 5): {scaler.scale_[:5]}")
    else:
        tqdm.write(f"Data min (first 5): {scaler.data_min_[:5]}")
        tqdm.write(f"Data max (first 5): {scaler.data_max_[:5]}")

    return scaler
#---------- parallel version of set_scaler (safe + fast) ─────────────────────────────────────────────────
import os
import numpy as np
from joblib import Parallel, delayed, dump
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import re

def _drop_time_if_present(df):
    time_cols = [c for c in df.columns if c.lower() == "time"]
    return df.drop(columns=time_cols) if time_cols else df

def _worker_patient_stats(patient_id: str, target_length: int, num_kinematic_var: int):
    """
    Computes (n, sum, sumsq) over ALL normalized samples for this patient across all trials.
    Returns None if patient not found or no data.
    """
    found_any = False
    n_total = 0
    sum_total = np.zeros((num_kinematic_var,), dtype=np.float64)
    sumsq_total = np.zeros((num_kinematic_var,), dtype=np.float64)

    for group_code, base_folder in base_folders.items():
        if patient_id not in list_patient_ids(base_folder):
            continue

        patient_folder = os.path.join(base_folder, patient_id, "trimmed_sc")
        trials_list, _ = load_patient_data(patient_folder, patient_id, group_code, verbose=False)

        found_any = True
        if not trials_list:
            return None

        for trial_df in trials_list:
            trial_df = _drop_time_if_present(trial_df)

            kin_df = trial_df.iloc[:, :num_kinematic_var]
            if kin_df.shape[1] != num_kinematic_var:
                # schema mismatch -> skip trial
                continue

            cycles = temporal_normalization_GC(kin_df, target_length=target_length)
            if cycles.size == 0:
                continue

            X = cycles.reshape(-1, num_kinematic_var).astype(np.float64, copy=False)
            n = X.shape[0]

            n_total += n
            sum_total += X.sum(axis=0)
            sumsq_total += (X * X).sum(axis=0)

        return (n_total, sum_total, sumsq_total)

    if not found_any:
        return None

    return (n_total, sum_total, sumsq_total)

def fit_standard_scaler_parallel(
    training_patient_ids: list[str],
    target_length: int,
    num_kinematic_var: int,
    scaler_filename: str = "global_kinematic_scaler_zscore_NoTime.pkl",
    n_jobs: int = 16,
):
    """
    Fits a StandardScaler using parallel aggregation of sums/sumsq (safe + fast).
    """
    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
        delayed(_worker_patient_stats)(pid, target_length, num_kinematic_var)
        for pid in tqdm(training_patient_ids, desc="Patients", unit="pt")
    )

    # Reduce
    n_total = 0
    sum_total = np.zeros((num_kinematic_var,), dtype=np.float64)
    sumsq_total = np.zeros((num_kinematic_var,), dtype=np.float64)

    used = 0
    for r in results:
        if r is None:
            continue
        n, s, ss = r
        if n <= 0:
            continue
        n_total += n
        sum_total += s
        sumsq_total += ss
        used += 1

    if n_total == 0:
        raise ValueError("No data processed: cannot fit scaler.")

    mean = sum_total / n_total
    var = (sumsq_total / n_total) - (mean ** 2)
    # numerical guard
    var = np.maximum(var, 1e-12)
    scale = np.sqrt(var)

    scaler = StandardScaler()
    scaler.mean_ = mean.astype(np.float64)
    scaler.var_ = var.astype(np.float64)
    scaler.scale_ = scale.astype(np.float64)
    scaler.n_features_in_ = num_kinematic_var
    scaler.n_samples_seen_ = n_total

    dump(scaler, scaler_filename)
    print(f"[INFO] StandardScaler saved to: {scaler_filename}")
    print(f"[INFO] Patients used: {used}/{len(training_patient_ids)} | total samples: {n_total}")
    print(f"[INFO] mean[:5]={scaler.mean_[:5]}")
    print(f"[INFO] scale[:5]={scaler.scale_[:5]}")

    return scaler

# ─── Dataset split  ─────────────────────────────────────────────────
def split_subjects_by_cycle_count(subjects_dict, train_ratio=0.7,val_ratio=0.15, seed=42):
    """
    Splits a dictionary of subjects and their cycle counts into three sets:
    training,validation and test, maintaining the proportion of cycles.
    The split uses a fixed seed which produces the same split every time.
    Later a cross-validation can be performed on the training set.

    Parameters:
        subjects_dict (dict): {'S001': cycles, ...}
        train_ratio (float): proportion for training (default: 0.7)
        val_ratio (float): proportion for validation (default: 0.15)
        seed (int): seed for reproducible randomness

    Returns:
        (train_subjects,val_subjects,test_subjects, total_cycles)
    """
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert train_ratio + val_ratio < 1

    random.seed(seed)
    subjects = list(subjects_dict.items())
    random.shuffle(subjects)

    total_cycles = sum(cycles for _, cycles in subjects)
    train_target = train_ratio * total_cycles
    val_target   = val_ratio   * total_cycles

    train_subjects = {}
    val_subjects   = {}
    test_subjects = {}
    accum_cycles = 0

    for subject, cycles in subjects:
        if accum_cycles < train_target:
            train_subjects[subject] = cycles
        elif accum_cycles < train_target + val_target:
            val_subjects[subject] = cycles
        else:
            test_subjects[subject] = cycles

        accum_cycles += cycles

    return train_subjects, val_subjects, test_subjects, total_cycles

# ─── Walking direction  ─────────────────────────────────────────────────
def _normalize_subject_id_for_trial_chars(pid: str, group_code: str) -> tuple[str, str]:
    """
    Returns (primary_id, fallback_id) for matching trial-characteristics 'id' column.
    - G2/G3: usually "SXXX"
    - G1: usually numeric "X"/"XX"
    """
    pid_str = str(pid).strip()
    m = re.search(r"(\d+)", pid_str)
    if not m:
        return (pid_str, pid_str)

    pid_int = int(m.group(1))
    sid_S = f"S{pid_int:03d}"   # "S039"
    sid_num = str(pid_int)      # "39"

    g = str(group_code).upper().strip()
    if g in ("G2", "G3"):
        return (sid_S, sid_num)
    if g == "G1":
        return (sid_num, sid_S)

    return (sid_S, sid_num)

def _digits_str(x: str) -> str:
    m = re.search(r"(\d+)", str(x))
    return str(int(m.group(1))) if m else str(x).strip()

def _pid_to_trialchar_id(pid: str, group_code: str) -> str:
    """
    G01: trial characteristics 'id' is numeric (e.g., 6)
    G02/G03: trial characteristics 'id' is 'SXXX' (e.g., S006)
    """
    g = str(group_code).upper().strip()
    pid_num = _digits_str(pid)      # "S006" -> "6"
    pid_int = int(pid_num)
    pid_S = f"S{pid_int:03d}"       # "S006"

    if g == "G01":
        return pid_num              # "6"
    else:
        return pid_S                # "S006"


def _load_trial_characteristics(group_code: str) -> pd.DataFrame:
    path = os.path.abspath(os.path.join(TRIAL_CHAR_DIR, f"{group_code}_GaitPrint_Trial_Characteristics.csv"))

    if path in _TRIAL_CHAR_CACHE:
        return _TRIAL_CHAR_CACHE[path]

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Trial characteristics CSV not found: {path}")

    df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"id", "day", "block", "trial", "direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {os.path.basename(path)}: {sorted(missing)}")

    for col in ("id", "day", "block", "trial", "direction"):
        df[col] = df[col].astype(str).str.strip()

    # Normalize day/block/trial: "D01" -> "1", "01" -> "1"
    df["day"] = df["day"].apply(_digits_str)
    df["block"] = df["block"].apply(_digits_str)
    df["trial"] = df["trial"].apply(_digits_str)

    # direction to canonical
    df["direction"] = df["direction"].str.lower()

    _TRIAL_CHAR_CACHE[path] = df
    return df

def get_walking_direction(pid: str, group_code: str, day: str, block: str, trial: str) -> str | None:
    df = _load_trial_characteristics(group_code)

    sid = _pid_to_trialchar_id(pid, group_code)
    day_k = _digits_str(day)       # "D01" -> "1"
    block_k = _digits_str(block)   # "B02" -> "2"
    trial_k = _digits_str(trial)   # "T03" -> "3"

    m = df[
        (df["id"] == sid) &
        (df["day"] == day_k) &
        (df["block"] == block_k) &
        (df["trial"] == trial_k)
    ]

    if m.empty:
        return None

    return m.iloc[0]["direction"]

def direction_to_numeric(direction: str | None) -> float:
    if direction is None:
        return np.nan
    d = str(direction).strip().lower()
    if d == "clockwise":
        return 1.0
    if d == "counterclockwise":
        return 0.0
    return np.nan


# ─── Tensor format  ─────────────────────────────────────────────────
def _encode_metadata_to_numeric(meta_dict: dict) -> np.ndarray:
    """
    Converts metadata from formats like 'S039', 'G01', 'D02', 'B03', 'T03'
    to integer numerical values [39, 1, 2, 3, 3].
    If meta_dict has 'direction_num', appends it.
    
    Args:
        meta_dict: Dictionary with keys:
            - 'patient_id': str, e.g., 'S039'
            - 'group': str,      e.g., 'G01'
            - 'day': str,        e.g., 'D02'
            - 'block': str,      e.g., 'B03'
            - 'trial': str,      e.g., 'T03'
    
    Returns:
        np.ndarray: Array of shape (5,) with the corresponding numerical values.
    """
    numeric_values = []
    for key in ["patient_id", "group", "day", "block", "trial"]:
        value = str(meta_dict[key])
        digits = int(re.sub(r"\D", "", value))
        numeric_values.append(digits)

    if "direction_num" in meta_dict:
        numeric_values.append(float(meta_dict["direction_num"]))

    return np.array(numeric_values, dtype=np.float32)



def purge_preprocessed_folder(patient_base: str,
                              subfolder_name: str = "preprocessed",
                              pattern: str = "*_preprocessed.npy",
                              verbose: bool = True) -> int:
    """
    Deletes old preprocessed .npy files inside patient/<subfolder_name>.
    Returns number of deleted files.
    """
    folder = os.path.join(patient_base, subfolder_name)
    if not os.path.isdir(folder):
        return 0

    files = glob.glob(os.path.join(folder, pattern))
    n = 0
    for f in files:
        try:
            os.remove(f)
            n += 1
        except OSError:
            pass

    if verbose and n > 0:
        print(f"[CLEAN] Removed {n} files from {folder} ({pattern})")
    return n


def save_patient_preprocessed_tensors(
    pid: str,
    group_code: str,
    target_length: int,
    num_kinematic_variables: int,
    global_scaler,
    subfolder_name: str = "preprocessed",
    verbose: bool = True,
    debug_first_trials: int = 0,
    debug_first_cycles: int = 0,
    purge_old: bool = False,
):
    """
    Processes all trimmed_sc trials for a patient and saves per-trial tensors as .npy.

    Steps:
    1) Load & clean CSV.
    2) Drop 'time' column (case-insensitive).
    3) Get trial-level walking direction; skip if missing/unknown.
    4) Temporal normalize cycles (segmentation inside temporal_normalization_GC).
    5) Feature-scale with global_scaler.
    6) Append metadata columns (direction + cycle_in_trial).
    7) Atomic save .npy + write a .done.json manifest for resumability.

    Metadata columns appended (order):
      [patient_id, group, day, block, trial, direction, cycle_in_trial]

    Resumability:
      If both <trial>_preprocessed.npy and <trial>.done.json exist and match expected dims,
      the trial is skipped (safe resume after interruptions).

    Args:
        pid: Patient identifier (e.g., "S039").
        group_code: Group code (must exist in base_folders).
        target_length: Number of timesteps per cycle after temporal normalization.
        num_kinematic_variables: Number of kinematic variables AFTER dropping time.
        global_scaler: Fitted scaler with .transform().
        subfolder_name: Output subfolder under patient.
        verbose: Print progress.
        debug_first_trials: Print debug info for first N processed trials per subject.
        debug_first_cycles: Print first N cycle_in_trial ids for debug trials.
        purge_old: If True, deletes old *_preprocessed.npy in output folder before processing.
    """
    import os
    import json
    import time
    import numpy as np

    if group_code not in base_folders:
        raise ValueError(f"Group '{group_code}' not found in base_folders.")

    patient_base = os.path.join(base_folders[group_code], pid)
    out_folder = os.path.join(patient_base, subfolder_name)
    os.makedirs(out_folder, exist_ok=True)

    trimmed_folder = os.path.join(patient_base, "trimmed_sc")
    if not os.path.isdir(trimmed_folder):
        if verbose:
            print(f"[WARN] No trimmed data for {pid} in {trimmed_folder}. Skipping.")
        return

    # Optional: delete old outputs (only once per patient run)
    if purge_old:
        purge_preprocessed_folder(patient_base, subfolder_name=subfolder_name, verbose=verbose)

    if verbose:
        print(f"[INFO] Saving preprocessed tensors to {out_folder}")

    def _trial_paths(pid_, day_, block_, trial_):
        base = os.path.join(out_folder, f"{pid_}_{day_}_{block_}_{trial_}")
        npy_path  = base + "_preprocessed.npy"
        tmp_path  = base + "_preprocessed.tmp.npy"
        done_path = base + ".done.json"
        return npy_path, tmp_path, done_path

    def _is_trial_done(npy_path: str, done_path: str, expected_cdim: int) -> bool:
        if not (os.path.isfile(npy_path) and os.path.isfile(done_path)):
            return False
        try:
            arr = np.load(npy_path, mmap_mode="r")
            if arr.ndim != 3:
                return False
            if arr.shape[1] != target_length:
                return False
            if arr.shape[2] != expected_cdim:
                return False
            with open(done_path, "r") as f:
                _ = json.load(f)
            return True
        except Exception:
            return False

    debug_trials_done = 0

    # Expected final channel dimension: features + meta(=7)
    expected_cdim = int(num_kinematic_variables) + 7

    # --- CHANGE 1: initialize final_path to avoid "referenced before assignment" ---
    final_path = None

    for day, block, trial, filepath in iter_trial_paths(trimmed_folder, pid, group_code):
        try:
            if not os.path.isfile(filepath):
                if verbose:
                    print(f"  [WARN] Missing file: {filepath}")
                continue

            # Decide whether to print debug info for this trial (only for the first N processed trials)
            do_debug = (debug_first_trials > 0) and (debug_trials_done < debug_first_trials)

            # Resume check (skip already completed trials)
            final_path, tmp_path, done_path = _trial_paths(pid, day, block, trial)
            if _is_trial_done(final_path, done_path, expected_cdim=expected_cdim):
                if verbose:
                    print(f"  [SKIP] done: {os.path.basename(final_path)}")
                continue

            df_trial = load_and_clean_csv(filepath)
            if df_trial.empty:
                if verbose:
                    print(f"[SKIP] Raw trial discarded (NaNs) → {os.path.basename(filepath)}")
                continue

            if df_trial.isna().any().any():
                print(f"[ERROR] Still NaNs in {os.path.basename(filepath)}!")
                continue

            # --- Remove time column (IMPORTANT) ---
            time_cols = [c for c in df_trial.columns if c.lower() == "time"]
            if time_cols:
                df_trial = df_trial.drop(columns=time_cols)
                if do_debug and verbose:
                    print("  [DEBUG] Removed time column")

            # ---- Trial-level walking direction (skip early if missing/unknown) ----
            direction_str = get_walking_direction(pid, group_code, day, block, trial)
            direction_num = direction_to_numeric(direction_str)

            if do_debug and verbose:
                print(f"  [DEBUG] {pid} {day} {block} {trial} direction={direction_str} -> {direction_num}")

            if not np.isfinite(direction_num):
                if verbose:
                    print(f"[SKIP] direction not found/unknown for {pid} {day} {block} {trial} -> {direction_str}")
                continue

            # 1) Temporal normalization per cycle (segmentation happens inside)
            kin_df = df_trial.iloc[:, :num_kinematic_variables]
            if kin_df.shape[1] != num_kinematic_variables:
                if verbose:
                    print(f"  [SKIP] kinematic cols mismatch in {os.path.basename(filepath)} "
                          f"(got {kin_df.shape[1]}, expected {num_kinematic_variables})")
                continue

            tensor_cycles = temporal_normalization_GC(kin_df, target_length)
            if tensor_cycles.size == 0:
                if verbose:
                    print(f"  [WARN] No cycles in {filepath}")
                continue

            # 2) Feature normalization
            n_cycles, _, _ = tensor_cycles.shape
            flat = tensor_cycles.reshape(-1, num_kinematic_variables)
            scaled_flat = global_scaler.transform(flat)
            scaled_cycles = scaled_flat.reshape(n_cycles, target_length, num_kinematic_variables).astype(np.float32, copy=False)

            if do_debug and verbose:
                print(f"  [DEBUG] n_cycles={n_cycles}")

            # 3) Metadata encoding (base meta per trial) + cycle_in_trial (per cycle)
            base_meta = _encode_metadata_to_numeric({
                "patient_id": pid,
                "group": group_code,
                "day": day,
                "block": block,
                "trial": trial,
                "direction_num": direction_num,
            })  # shape: (6,) because direction_num appended

            cycle_in_trial = np.arange(n_cycles, dtype=np.float32)[:, None]  # (n_cycles, 1)

            if do_debug and verbose and debug_first_cycles > 0:
                shown = cycle_in_trial[:debug_first_cycles, 0].astype(int).tolist()
                print(f"  [DEBUG] cycle_in_trial first {debug_first_cycles}/{n_cycles}: {shown}")

            base_meta_mat = np.tile(np.asarray(base_meta, dtype=np.float32)[None, :], (n_cycles, 1))  # (n_cycles, 6)
            cycle_meta = np.concatenate([base_meta_mat, cycle_in_trial], axis=1)  # (n_cycles, 7)

            meta_tensor = np.tile(cycle_meta[:, None, :], (1, target_length, 1))  # (n_cycles, T, 7)

            # 4) Concatenate features + metadata
            final_tensor = np.concatenate(
                (scaled_cycles, meta_tensor.astype(np.float32, copy=False)),
                axis=2,
            )

            # Sanity check
            if final_tensor.shape[2] != expected_cdim:
                if verbose:
                    print(f"  [ERROR] Unexpected c_dim for {os.path.basename(filepath)}: "
                          f"{final_tensor.shape[2]} (expected {expected_cdim})")
                continue

            # 5) Atomic save .npy
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

            np.save(tmp_path, final_tensor)
            os.replace(tmp_path, final_path)

            # 6) Write .done.json manifest
            payload = {
                "file": os.path.basename(final_path),
                "pid": pid,
                "group": group_code,
                "day": day,
                "block": block,
                "trial": trial,
                "shape": [int(x) for x in final_tensor.shape],
                "num_kinematic_variables": int(num_kinematic_variables),
                "meta_cols": int(cycle_meta.shape[1]),
                "direction_str": direction_str,
                "direction_num": float(direction_num),
                "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            }
            with open(done_path, "w") as f:
                json.dump(payload, f, indent=2)

            if verbose:
                print(f"  [OK] {os.path.basename(final_path)} -> shape {final_tensor.shape} (meta_cols={cycle_meta.shape[1]})")

            if do_debug:
                debug_trials_done += 1

        except Exception as e:
            # --- CHANGE 2: robust error message even if final_path not assigned ---
            name = os.path.basename(final_path) if final_path else os.path.basename(filepath)
            print(f"[ERROR] {pid}: {name}: {e}")
            continue

def clean_and_save_trial(input_path: str, eps: float = 1e-8) -> bool:
    """
    Load a preprocessed .npy trial, remove cycles containing NaN/Inf,
    and keep only files with valid data.

    Returns:
        True if at least one cycle remains and file is kept,
        False if all cycles were corrupt (file removed).
    """
    import os 
    arr = np.load(input_path).astype(np.float32)
    # Keep only fully finite cycles
    mask = np.isfinite(arr).all(axis=(1, 2))
    clean = arr[mask]
    if clean.size > 0:
        # Save cleaned data back to the same path
        np.save(input_path, clean)
        return True
    else:
        # Remove the file entirely if no valid cycles remain
        try:
            os.remove(input_path)
        except OSError:
            pass
        return False

def preprocess_all_groups_and_patients(
    target_length: int,
    num_kinematic_variables: int,
    global_scaler,
    subfolder_name: str = "preprocessed",
    verbose: bool = False,
    groups_to_process: list[str] | None = None,
    debug_first_trials=2,
    debug_first_cycles=5
):
    """
    Applies save_patient_preprocessed_tensors to every patient in every group,
    then cleans each generated .npy by removing cycles with NaN/Inf.

    Args:
        target_length: Timesteps per cycle for normalization.
        num_kinematic_variables: Number of biomechanical variables.
        global_scaler: Scaler trained on training set.
        subfolder_name: Subfolder name where preprocessed .npy are saved.
        verbose: Print progress.
    """
    import os

    for group_code, group_path in base_folders.items():
        if groups_to_process is not None and group_code not in groups_to_process:
            continue
        if verbose:
            print(f"\n[INFO] Processing group: {group_code}")

        # --- CHANGE 3: deterministic order ---
        patient_ids = sorted([d for d in os.listdir(group_path)
                              if os.path.isdir(os.path.join(group_path, d))])

        for pid in patient_ids:
            try:
                if verbose:
                    print(f"[INFO]  Patient: {pid}")
                # 1) Save raw preprocessed tensors for the patient (resumable by trial)
                save_patient_preprocessed_tensors(
                    pid=pid,
                    group_code=group_code,
                    target_length=target_length,
                    num_kinematic_variables=num_kinematic_variables,
                    global_scaler=global_scaler,
                    subfolder_name=subfolder_name,
                    verbose=verbose,
                    debug_first_trials=debug_first_trials,
                    debug_first_cycles=debug_first_cycles,
                    purge_old=False,
                )

                # 2) Clean each .npy in the subfolder
                pre_dir = os.path.join(group_path, pid, subfolder_name)
                if os.path.isdir(pre_dir):
                    for fname in os.listdir(pre_dir):
                        if fname.endswith("_preprocessed.npy"):
                            inp = os.path.join(pre_dir, fname)
                            ok = clean_and_save_trial(inp)
                            if not ok:
                                print(f"[CLEAN] Discarded corrupt file: {os.path.basename(inp)}")

            except Exception as e:
                print(f"[ERROR] Patient {pid}: {e}")