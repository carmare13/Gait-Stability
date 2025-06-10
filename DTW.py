import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dtaidistance import dtw
from dtaidistance.dtw import distance_matrix
from scipy.signal import decimate
from Data_loader    import base_folders, list_patient_ids, iter_trial_paths, load_patient_data
from segment_utils import segment_cycles_simple
from PP_pipeline import segment_downsamp
from downsample import downsample_df
from summary_utils  import ensure_dir

#Choose dtw_fabio for the complete distance matrices for advanced post‐analysis
#Choose dtw_di for summary statistics per trial and tidy tabular result
#Choose dtw_di2 for the complete distance matrices on the segmented without normalization or downsampling
#Choose dtw_not_segmented for pairwise DTW on full trials without segmentation
#Choose dtw_ns_normalized for pairwise DTW on full trials without segmentation, but normalized (z-score) 

def dtw_fabio(group_code,
              source='downsampled',
              output_base='DTW',
              verbose=False):
    """
    Compute and save DTW distance matrices for every patient/trial in `group_code`.
    
    - Reads CSVs from: <base>/<group_code>/<source>/<patient_id>/
    - Segments each trial into cycles via `segment_cycles()`
    - Computes full NxN DTW matrix inline:
        D[i][j] = dtw.distance(cycles[i].values, cycles[j].values)
    - Builds dtw_all = { patient_id: { trial_key: D as list of lists } }
    - Dumps dtw_all → JSON at: output_base/dtw_all.json
    
    Returns:
        dtw_all (dict)
    """
    base        = base_folders[group_code]
    data_folder = os.path.join(base, source)
    ensure_dir(output_base)

    dtw_all = {}
    for pid in tqdm(list_patient_ids(base),
                    desc=f"DTW {group_code}",
                    unit="patient"):
        dtw_all[pid] = {}
        patient_folder = os.path.join(data_folder, pid)

        for day, block, trial, path in iter_trial_paths(patient_folder, pid, group_code):
            if not os.path.isfile(path):
                if verbose:
                    print(f"[WARN] Missing file: {path}")
                continue

            # 1) load downsampled trial
            df_trial = pd.read_csv(path)
            # 2) segment into cycles
            cycles = segment_cycles(df_trial)
            if not cycles:
                if verbose:
                    print(f"[WARN] No cycles in {os.path.basename(path)}")
                continue

            # 3) compute NxN DTW matrix inline
            n = len(cycles)
            D = [[0.0]*n for _ in range(n)]
            for i in range(n):
                c1 = cycles[i].values
                for j in range(i+1, n):
                    c2 = cycles[j].values
                    dist = dtw.distance(c1, c2)
                    D[i][j] = D[j][i] = dist

            trial_key = f"{day}_{block}_{trial}"
            dtw_all[pid][trial_key] = D

            if verbose:
                print(f"[INFO] {pid} {trial_key}: computed {n}×{n} DTW matrix")

    # 4) save to JSON
    out_file = os.path.join(output_base, 'dtw_all.json')
    with open(out_file, 'w') as f:
        json.dump(dtw_all, f)

    if verbose:
        print(f"[OK] All DTW results written to {out_file}")

    return dtw_all

def dtw_di(group_code,
           signal_col        = 'Ankle Dorsiflexion RT (deg)',
           min_length        = 20,
           downsample_factor = 4,
           output_base       = 'DTW',
           verbose           = False):
    """
    For each patient/trial in `group_code`:
      1) loads raw CSVs
      2) segments and downsamples only `signal_col`
      3) computes pairwise DTW for each cycle-pair
      4) summarizes as mean, median, std, n_pairs
    Saves:
      - nested dict → output_base/dtw_all.json
      - flat stats → output_base/dtw_intra_trial_stats.csv/json
    Returns:
      pd.DataFrame with one row per trial
    """
    base_folder = base_folders[group_code]
    data_folder = base_folder  # Use raw always for events
    ensure_dir(output_base)

    dtw_all = {}
    records = []

    for pid in tqdm(list_patient_ids(base_folder), desc="Patients"):
        dtw_all[pid] = {}
        patient_folder = os.path.join(base_folder, pid) 
        dfs, paths = load_patient_data(patient_folder, pid, group_code, subfolder=None)
        if not dfs or not paths:
            if verbose:
                print(f"[WARN] No data for {pid}")
            continue
        for df_trial, filepath in zip(dfs, paths):
            trial_id = os.path.basename(filepath).replace('.csv','')

            # Segment and downsample the selected signal
            cycles = segment_downsamp(
                df_trial,
                signal_col=signal_col,
                min_length=min_length,
                downsample_factor=downsample_factor
            )
            n_cycles = len(cycles)

            # Compute DTW and stats
            if n_cycles > 1:
                dists = []
                for i in range(n_cycles):
                    c1 = cycles[i]
                    for j in range(i+1, n_cycles):
                        c2 = cycles[j]
                        dists.append(dtw.distance(c1, c2))
                arr = np.array(dists)
                stats = {
                    'mean': float(arr.mean()),
                    'median': float(np.median(arr)),
                    'std': float(arr.std()),
                    'n_pairs': int(len(arr))
                }
            else:
                stats = {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'n_pairs': 0}

            dtw_all[pid][trial_id] = stats
            records.append({
                'patient_id': pid,
                'trial_id':   trial_id,
                'n_cycles':   n_cycles,
                **stats
            })

            if verbose:
                print(f"[INFO] {pid} {trial_id}: cycles={n_cycles}, stats={stats}")

    # Save nested dict
    out_json_all = os.path.join(output_base, 'dtw_all.json')
    with open(out_json_all, 'w') as f:
        json.dump(dtw_all, f)

    df_results = pd.DataFrame.from_records(records)
    df_results = df_results[['patient_id','trial_id','n_cycles','n_pairs','mean','median','std']]
    csv_path = os.path.join(output_base, f'dtw_intra_trial_stats_{group_code}.csv')
    df_results.to_csv(csv_path, index=False)
    json_path = os.path.join(output_base, f'dtw_intra_trial_stats_{group_code}.json')
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
    if verbose:
        print(f"[OK] Nested DTW results → {out_json_all}")
        print(df_results.head())
    return df_results

def dtw_di2(group_code,
            signal_col        = 'Ankle Dorsiflexion RT (deg)',
            output_base       = 'DTW',
            verbose           = False):
    """
        Computes pairwise DTW for each cycle-pair within each trial,
        without normalizing the signal.
        Uses all dataframe columns for segmentation but only the specified
        column for DTW calculation.
    """
    base_folder = base_folders[group_code]
    ensure_dir(output_base)

    dtw_all = {}
    records = []

    for pid in tqdm(list_patient_ids(base_folder), desc="Patients"):
        dtw_all[pid] = {}
        patient_folder = os.path.join(base_folder, pid, "trimmed")
        dfs, paths = load_patient_data(patient_folder, pid, group_code, subfolder=None)
        if not dfs or not paths:
            if verbose:
                print(f"[WARN] No data para paciente {pid}")
            continue

        for df_trial, filepath in zip(dfs, paths):
            trial_id = os.path.basename(filepath).replace('.csv','')
            ciclos_df = segment_cycles_simple(
                df_trial,
                print_cycle_length=False
            )
            n_cycles = len(ciclos_df)
            if n_cycles > 0:
                ciclos_signal = [
                    ciclo_df[signal_col].values
                    for ciclo_df in ciclos_df
                ]
            else:
                ciclos_signal = []
            if n_cycles > 1:
                try:
                    D = dtw.distance_matrix_fast(ciclos_signal)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Fast DTW falló para {trial_id}: {e}. Usando versión lenta.")
                    D = dtw.distance_matrix(ciclos_signal, parallel=True)

                triu_indices = np.triu_indices(n_cycles, k=1)
                dists = D[triu_indices]
                stats = {
                    'mean':   float(np.mean(dists)),
                    'median': float(np.median(dists)),
                    'std':    float(np.std(dists)),
                    'n_pairs': int(len(dists))
                }
            else:
                stats = {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'n_pairs': 0}

            dtw_all[pid][trial_id] = stats
            records.append({
                'patient_id': pid,
                'trial_id':   trial_id,
                'n_cycles':   n_cycles,
                **stats
            })

            if verbose:
                print(f"[INFO] {pid} {trial_id}: ciclos={n_cycles}, stats={stats}")

    out_json_all = os.path.join(output_base, 'dtw_all_cdist.json')
    with open(out_json_all, 'w') as f:
        json.dump(dtw_all, f)

    df_results = pd.DataFrame.from_records(records)
    df_results = df_results[['patient_id','trial_id','n_cycles','n_pairs','mean','median','std']]
    csv_path = os.path.join(output_base, f'dtw_intra_trial_stats_cdist_{group_code}.csv')
    df_results.to_csv(csv_path, index=False)
    json_path = os.path.join(output_base, f'dtw_intra_trial_stats_cdist_{group_code}.json')
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)

    if verbose:
        print(f"[OK] Nested DTW results → {out_json_all}")
        print(df_results.head())

    return df_results

def dtw_not_segmented(
    group_code,
    signal_col='Ankle Dorsiflexion RT (deg)',
    output_base='DTW',
    verbose=False
):
    """
    Compute pairwise DTW distances between entire trials (not segmented) for each patient    
    Saves results in JSON (nested dict) and CSV (flat table).
    
    Returns:
        dtw_all: dict { patient_id: { (trial_i, trial_j): distance } }
    """
    
    base_folder = base_folders[group_code]
    ensure_dir(output_base)

    dtw_all = {}
    records = []

    for pid in tqdm(list_patient_ids(base_folder), desc="Patients"):
        dtw_all[pid] = {}

        dfs, paths = load_patient_data(os.path.join(base_folder, pid), pid, group_code, subfolder=None)
        if not dfs or not paths:
            if verbose:
                print(f"[WARN] No data for {pid}")
            continue

        signals = []
        trial_ids = []

        for df, path in zip(dfs, paths):
            trial_id = os.path.basename(path).replace('.csv', '')
            if signal_col not in df.columns:
                if verbose:
                    print(f"[WARN] Signal {signal_col} not found in {trial_id}")
                continue
            try:
                sig = df[signal_col].values.astype(float)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Error loading signal {signal_col} from {trial_id}: {e}")
                continue
            

            sig = np.asarray(sig).flatten()
            if sig.size == 0:
                if verbose:
                    print(f"[WARN] Empty signal in {trial_id}")
                continue


            signals.append(sig)

            trial_ids.append(trial_id)

        n_trials = len(signals)
        if n_trials < 2:
            if verbose:
                print(f"[INFO] Not enough trials for patient {pid} to compute DTW")
            continue

        # Compute pairwise DTW distance matrix for all full trials
        try:
            D = dtw.distance_matrix_fast(signals)
        except Exception as e:
            if verbose:
                print(f"[WARN] Fast DTW failed for patient {pid}: {e}. Falling back to slow DTW.")
            D = dtw.distance_matrix(signals, parallel=True)

        # Extract upper triangle indices (i<j)
        triu_indices = np.triu_indices(n_trials, k=1)
        for i, j in zip(*triu_indices):
            dist = float(D[i, j])
            dtw_all[pid][f"{trial_ids[i]}_{trial_ids[j]}"] = dist

            records.append({
                'patient_id': pid,
                'trial_i': trial_ids[i],
                'trial_j': trial_ids[j],
                'dtw_distance': dist
            })

            #if verbose:
                #print(f"[INFO] {pid} {trial_ids[i]} vs {trial_ids[j]}: DTW={dist:.3f}")

    # Save nested dict JSON
    out_json_path = os.path.join(output_base, f'dtw_not_segmented_{group_code}.json')
    with open(out_json_path, 'w') as f_json:
        json.dump(dtw_all, f_json)

    # Save flat CSV
    df_records = pd.DataFrame.from_records(records) 
    out_csv_path = os.path.join(output_base, f'dtw_not_segmented_{group_code}.csv')
    df_records.to_csv(out_csv_path, index=False)

    if verbose:
        print(f"[OK] Saved DTW results JSON to {out_json_path}")
        print(f"[OK] Saved DTW results CSV to {out_csv_path}")

    return dtw_all

def dtw_ns_normalized(
    group_code,
    signal_col='Ankle Dorsiflexion RT (deg)',
    output_base='DTW',
    verbose=False
):
    """
    Compute pairwise DTW distances between entire trials (not segmented) for each patient,
    after normalizing (z-score) each trial's signal.    
    Saves results in JSON (nested dict) and CSV (flat table).
    
    Returns:
        dtw_all: dict { patient_id: { "trial_i_trial_j": distance } }
    """

    base_folder = base_folders[group_code]
    ensure_dir(output_base)

    dtw_all = {}
    records = []

    for pid in tqdm(list_patient_ids(base_folder), desc="Patients"):
        dtw_all[pid] = {}

        dfs, paths = load_patient_data(os.path.join(base_folder, pid), pid, group_code, subfolder=None)
        if not dfs or not paths:
            if verbose:
                print(f"[WARN] No data for {pid}")
            continue

        signals = []
        trial_ids = []

        for df, path in zip(dfs, paths):
            trial_id = os.path.basename(path).replace('.csv', '')
            if signal_col not in df.columns:
                if verbose:
                    print(f"[WARN] Signal {signal_col} not found in {trial_id}")
                continue
            try:
                sig = df[signal_col].values
                sig = np.asarray(sig)
                if sig.ndim > 1:
                    sig = sig.flatten()
                sig = sig.astype(float)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Error loading signal {signal_col} from {trial_id}: {e}")
                continue
             

            # Normalize with z-score
            sig = np.asarray(sig)
            if sig.ndim > 1:
                sig = sig.flatten()

            sig_mean = np.mean(sig)
            sig_std = np.std(sig)

            if sig_std > 0:
                sig = (sig - sig_mean) / sig_std
            else:
                sig = sig - sig_mean

            
            if sig.size == 0:
                if verbose:
                    print(f"[WARN] Empty signal {trial_id}")
                continue

            signals.append(sig)
            trial_ids.append(trial_id)

        n_trials = len(signals)
        if n_trials < 2:
            if verbose:
                print(f"[INFO] Not enough trials for patient {pid} to compute DTW")
            continue

        # Compute pairwise DTW distance matrix for all full trials
        try:
            D = dtw.distance_matrix_fast(signals)
        except Exception as e:
            if verbose:
                print(f"[WARN] Fast DTW failed for patient {pid}: {e}. Falling back to slow DTW.")
            D = dtw.distance_matrix(signals, parallel=True)

        # Extract upper triangle indices (i<j)
        triu_indices = np.triu_indices(n_trials, k=1)
        for i, j in zip(*triu_indices):
            dist = float(D[i, j])
            dtw_all[pid][f"{trial_ids[i]}_{trial_ids[j]}"] = dist

            records.append({
                'patient_id': pid,
                'trial_i': trial_ids[i],
                'trial_j': trial_ids[j],
                'dtw_distance': dist
            })

            if verbose:
                print(f"[INFO] {pid} {trial_ids[i]} vs {trial_ids[j]}: DTW={dist:.3f}")

    # Save nested dict JSON
    out_json_path = os.path.join(output_base, f'dtw_ns_normalized_{group_code}.json')
    with open(out_json_path, 'w') as f_json:
        json.dump(dtw_all, f_json)

    # Save flat CSV
    df_records = pd.DataFrame.from_records(records)
    out_csv_path = os.path.join(output_base, f'dtw_ns_normalized_{group_code}.csv')
    df_records.to_csv(out_csv_path, index=False)

    if verbose:
        print(f"[OK] Saved DTW normalized results JSON to {out_json_path}")
        print(f"[OK] Saved DTW normalized results CSV to {out_csv_path}")

    return dtw_all



