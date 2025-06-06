import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

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
from segment_utils import segment_cycles_simple
from sklearn.preprocessing import StandardScaler  # Z-score normalization
from sklearn.preprocessing import MinMaxScaler  # Min-Max normalization
from joblib import dump, load  # To save and load the scaler



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
        segs = get_segments_for_patient(pid, group_code, source, verbose)
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
    from scipy.interpolate import interp1d
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
    ):
    """
    This script builds a single global scaler to normalize all kinematic features 
    Locates and loads that patient’s preprocessed (“trimmed”) trial data.
    Applies a temporal normalization routine (temporal_normalization_GC)
    Flattens the resulting 3D tensor into a 2D array and incrementally fits a StandardScaler 
    Once all six patients have been processed, the scaler has learned the global
    mean and standard deviation for every kinematic variable. The script then:
    Verifies that the scaler was successfully fitted,
    Saves the trained scaler and
    Prints out the first few entries of the learned means and standard deviations for sanity checking.
    """
    scaler_type = scaler_type.lower()
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type has to be'standard' or 'minmax'.")
    for patient_id in training_patient_ids:
        found_patient = False
        for group_code, base_folder in base_folders.items():
            patient_ids_in_group = list_patient_ids(base_folder)
            if patient_id in patient_ids_in_group:
                patient_folder = os.path.join(base_folder, patient_id, "trimmed")
                print(f"Processing patient {patient_id} (group {group_code})...")                
                
                dfs_trials, _ = load_patient_data(patient_folder, patient_id, group_code, verbose=True)

                if not dfs_trials:
                    print(f"  [INFO] No data found for patient {patient_id} in {patient_folder}.")
                    continue
                
                for trial_df in dfs_trials:
                    kinematic_data_df = trial_df.iloc[:, :num_kinematic_var]

                    # Apply temporal normalization
                    normalized_cycles_3d_tensor = temporal_normalization_GC(kinematic_data_df, target_length=target_length)

                    if normalized_cycles_3d_tensor.size > 0:
                        # Flatten the tensor to 2D for partial_fit
                        # Shape: (num_cycles_this_trial * target_length, NUM_KINEMATIC_VARIABLES)
                        reshaped_cycles = normalized_cycles_3d_tensor.reshape(-1, num_kinematic_var)
                        
                        # --- Incremental Fitting ---
                        scaler.partial_fit(reshaped_cycles)
                        print(f"    [OK] Processed trial for {patient_id} ({trial_df['day'].iloc[0]}_{trial_df['block'].iloc[0]}_{trial_df['trial'].iloc[0]}) - Scaler updated.")
                    else:
                        print(f"    [WARN] No valid cycles obtained from trial for {patient_id} ({trial_df['day'].iloc[0]}_{trial_df['block'].iloc[0]}_{trial_df['trial'].iloc[0]}).")
                
                found_patient_in_group = True
                break # Exit the group loop once the patient has been found and processed
        
        if not found_patient_in_group:
            print(f"[ERROR] Patient '{patient_id}' not found in any of the specified group folders.")

    try:
        _ = scaler.mean_ 
        print("\nScaler fitting process complete.")
    except AttributeError:
        raise ValueError("The scaler was not fitted. No data was processed. ")

    # --- 3. Save the Fitted Scaler ---
    dump(scaler, scaler_filename)
    print(f"Scaler saved to: {scaler_filename}")

    # --- 4. Verification 
    if isinstance(scaler, StandardScaler):
        print(f"\nMean of each variable (first 5): {scaler.mean_[:5]}")
        print(f"Standard deviation of each variable (first 5): {scaler.scale_[:5]}")
    else:  # MinMaxScaler
        print(f"\nFirst 5 data_min: {scaler.data_min_[:5]}")
        print(f"First 5 data_max: {scaler.data_max_[:5]}")

    return scaler   

# ─── Tensor format  ─────────────────────────────────────────────────

def save_patient_preprocessed_tensors(pid, group_code, target_length, subfolder_name: str = "preprocessed", verbose = True):
    """
    Processes all trials for a patient:
    1. Temporally normalizes each cycle using temporal_normalization_GC
    2. Applies feature normalization using the global_scaler.
    3. Appends numerical representations of identification columns.
    4. Saves each resulting 3D tensor as .npy files within a patient subfolder.

    Args:
        pid (str): Patient identifier.
        group_code (str): Code of the group to which the patient belongs (e.g., 'G01', 'G03').
        target_length (int): Fixed number of desired samples per cycle after temporal normalization.
        subfolder_name (str, optional): Name of the subfolder where the tensors will be saved (default is "preprocessed").
        verbose (bool): If True, prints progress messages.
    """
    # Check that group_code exists in base_folders
    if group_code not in base_folders:
        raise ValueError(f"The group_code '{group_code}' does not exist in base_folders.")
    
    # Construct patient base folder path
    e_patient_folder = os.path.join(base_folders[group_code], pid)
    patient_data_folder = os.path.join(e_patient_folder, "trimmed") 
    
    if not os.path.isdir(patient_data_folder):
        print(f"[WARN] Patient data folder '{patient_data_folder}' does not exist. Skipping patient {pid}.")
        return # Return instead of raising error to allow other patients to process

    # Create the subfolder for preprocessed tensors
    tensors_folder = os.path.join(e_patient_folder, subfolder_name)
    os.makedirs(tensors_folder, exist_ok=True)
    if verbose:
        print(f"[INFO] Saving preprocessed tensors in: {tensors_folder}")

    # Iterate over each trial of the patient
    for day, block, trial, filepath in iter_trial_paths(patient_data_folder, pid, group_code):
        if not os.path.isfile(filepath):
            if verbose:
                print(f"  [WARN] Trial file not found: {filepath}")
            continue

        try:
            # Load the trial DataFrame
            df_trial = pd.read_csv(filepath, dtype=float) # Ensure float dtype for kinematic data

            if df_trial.empty:
                if verbose:
                    print(f"  [WARN] Empty file: {filepath}. Skipping.")
                continue

            # Extract kinematic data and metadata
            kinematic_data_df = df_trial.iloc[:, :NUM_KINEMATIC_VARIABLES]
            # Ensure metadata columns exist; if not, you might need to add them in load_patient_data
            # For this example, assuming 'patient_id', 'group', 'day', 'block', 'trial' are added by load_patient_data
            # if they were present in the original CSVs. If not, they'd be derived from the loop variables.
            # Here, we'll get them from the function parameters/loop context directly.
            
            # --- Temporal Normalization ---
            normalized_kinematic_3d_tensor = temporal_normalize_gait_cycles(kinematic_data_df, target_length)

            if normalized_kinematic_3d_tensor.size == 0:
                if verbose:
                    print(f"  [WARN] No cycles detected in: {filepath}. Skipping.")
                continue

            # --- Feature Normalization using Global Scaler ---
            # Reshape from (N_cycles, T_length, N_kinematic_vars) to (N_samples_total, N_kinematic_vars)
            num_cycles_in_trial, _, _ = normalized_kinematic_3d_tensor.shape
            reshaped_for_scaler = normalized_kinematic_3d_tensor.reshape(-1, NUM_KINEMATIC_VARIABLES)
            
            # Apply the already fitted global scaler
            scaled_kinematic_2d = global_scaler.transform(reshaped_for_scaler)
            
            # Reshape back to 3D: (N_cycles, T_length, N_kinematic_vars)
            scaled_kinematic_3d_tensor = scaled_kinematic_2d.reshape(num_cycles_in_trial, target_length, NUM_KINEMATIC_VARIABLES)

            # --- Append Identification Columns ---
            # Convert current trial's metadata to numeric representation
            # This 'metadata_series' needs to be built from the context variables
            current_trial_metadata = pd.Series({
                'patient_id': pid,
                'group': group_code,
                'day': day,
                'block': block,
                'trial': trial
            })
            encoded_id_values = _encode_metadata_to_numeric(current_trial_metadata)

            # Replicate ID values to match the shape (num_cycles, target_length, num_id_cols)
            # This creates a tensor where each ID value is repeated for each time step
            num_id_columns = len(encoded_id_values)
            replicated_id_tensor = np.tile(encoded_id_values, (num_cycles_in_trial, target_length, 1))
            
            # Concatenate scaled kinematic data with replicated ID data along the last axis (features)
            # Resulting shape: (num_cycles, target_length, NUM_KINEMATIC_VARIABLES + num_id_columns)
            final_preprocessed_tensor = np.concatenate(
                (scaled_kinematic_3d_tensor, replicated_id_tensor),
                axis=2
            )

            # Construct filename for the tensor
            filename = f"{pid}_{day}_{block}_{trial}_preprocessed.npy"
            save_path = os.path.join(tensors_folder, filename)

            # Save the 3D tensor as a .npy file
            np.save(save_path, final_preprocessed_tensor)

            if verbose:
                print(f"  [OK] Saved preprocessed tensor for trial {day}-{block}-{trial}: {filename} (shape={final_preprocessed_tensor.shape})")

        except Exception as e:
            print(f"  [ERROR] Processing '{filepath}': {e}")

def preprocess_all_groups_and_patients(target_length: int, subfolder_name: str = "preprocessed_tensors", verbose: bool = True):
    """
    Iterates over all groups and patients in the database, processes all trials,
    and saves the preprocessed tensors.

    Args:
        target_length (int): Fixed number of desired samples per cycle for temporal normalization.
        subfolder_name (str, optional): Name of the subfolder where the tensors will be saved (default "preprocessed_tensors").
        verbose (bool): If True, prints progress messages.
    """
    print("\n--- Starting full preprocessing of all data ---")
    for group_code, base_folder in base_folders.items():
        if verbose:
            print(f"\n[INFO] Processing group: {group_code}")
        
        patient_ids = list_patient_ids(base_folder)
        if not patient_ids:
            if verbose:
                print(f"  [WARN] No patients found in group '{group_code}'.")
            continue

        for pid in patient_ids:
            if verbose:
                print(f"\n[INFO] Processing patient: {pid} from group {group_code}")
            try:
                save_patient_preprocessed_tensors(
                    pid=pid, group_code=group_code, 
                    target_length=target_length, 
                    subfolder_name=subfolder_name, 
                    verbose=verbose
                )
            except Exception as e:
                print(f"[ERROR] Error processing patient '{pid}': {e}")
    print("\n--- Full preprocessing complete ---")