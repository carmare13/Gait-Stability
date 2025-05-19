import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dtaidistance import dtw
from scipy.signal import decimate
from Data_loader    import base_folders, list_patient_ids, iter_trial_paths, load_patient_data
from pipeline import segment_downsamp
from summary_utils  import ensure_dir


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
    csv_path = os.path.join(output_base, 'dtw_intra_trial_stats.csv')
    df_results.to_csv(csv_path, index=False)
    json_path = os.path.join(output_base, 'dtw_intra_trial_stats.json')
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2)
    if verbose:
        print(f"[OK] Nested DTW results → {out_json_all}")
        print(df_results.head())
    return df_results

#Choose dtw_fabio for the complete distance matrices for advanced post‐analysis
#Choose dtw_di for summary statistics per trial and tidy tabular result
