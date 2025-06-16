import os
import pandas as pd
from itertools import product
import json
from typing import Optional

# ─── Configuration ────────────────────────────────────────────────────────────

# days, blocks and trials per patient (variable names in lowercase)
days   = ["D01", "D02"]
blocks = ["B01", "B02", "B03"]
trials = ["T01", "T02", "T03"]

# mapping of group code to its base folder
#project_root = os.path.dirname(os.path.abspath(__file__))
project_root = "/mnt/storage/dmartinez" #Now the database is in the server
base_folders = {
    "G01": os.path.join(project_root, "young adults (19–35 years old)"),
    "G03": os.path.join(project_root, "old adults (56+ years old)")
}

# root folder for all EDA outputs (keep uppercase)
output_root = os.path.join(".", "EDA")


# ─── Utility Functions ────────────────────────────────────────────────────────

def ensure_dir(path):
    """Create the directory at `path` if it does not already exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def list_patient_ids(base_folder):
    """
    Return a sorted list of all patient subfolder names in `base_folder`.
    Only folders named 'Sxxx' (4-chars starting with 'S') are included.
    """
    return sorted(
        name for name in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, name))
           and name.startswith("S")
           and len(name) == 4
    )


# ─── Data Loading ─────────────────────────────────────────────────────────────
def iter_trial_paths(patient_folder, patient_id, group_code):
    """
    Yield one trial at a time as (day, block, trial, full_path).
    Does NOT read the file, only constructs the expected path.
    """
    for d, b, t in product(days, blocks, trials):
        fname = f"{patient_id}_{group_code}_{d}_{b}_{t}.csv"
        yield d, b, t, os.path.join(patient_folder, fname)

def load_and_clean_csv(path: str) -> Optional[pd.DataFrame]:
    df = pd.read_csv(path)
    df.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)
    df.ffill(axis=0, inplace=True)
    df.bfill(axis=0, inplace=True)
    if df.isna().any().any():
        print(f"[SKIP] Discarded trial {os.path.basename(path)} due to remaining NaNs")
        return None
    return df


def load_patient_data(
    patient_folder: str,
    patient_id: str,
    group_code: str,
    subfolder: str | None = None,
    verbose: bool = False
) -> tuple[list[pd.DataFrame], list[str]]:
    """
    Read all available trial CSVs for one patient into a list of DataFrames.
    Adds metadata columns: patient_id, group, day, block, trial.
    Returns:
        dfs (list of DataFrames), paths (list of str)
    """
    df_list: list[pd.DataFrame] = []
    file_list: list[str]       = []
    
    for d, b, t, path in iter_trial_paths(patient_folder, patient_id, group_code):
        if not os.path.isfile(path):
            if verbose:
                print(f"  [WARN] file not found: {path}")
            continue
        
        try:
            df = load_and_clean_csv(path)
            if df is None:
                continue
            if df.empty:
                if verbose:
                    print(f"[WARN] empty file: {path}")
                continue

            # Attach metadata
            df['patient_id'] = patient_id
            df['group']      = group_code
            df['day']        = d
            df['block']      = b
            df['trial']      = t

            df_list.append(df)
            file_list.append(path)

        except Exception as e:
            print(f"[ERROR] reading {path}: {e}")

    return (df_list, file_list) if df_list else ([], [])



def summarize_file(path, usecols=None, dtype=None):

    """
    Read a single CSV at `path`, compute descriptive stats and count missing values.
    Returns a DataFrame with the transpose of describe() and a 'n_missing' column.
    """
    df = pd.read_csv(path, usecols=usecols, dtype=dtype)
    summary = df.describe().transpose()
    summary['n_missing'] = df.isna().sum()
    return summary

# ─── AE  ─────────────────────────────────────────────────────────────
def load_subjects_from_json(json_path):
    """
    Reads a JSON file containing a dictionary of subjects and returns a list of their IDs.
    Expected format: { "S001": {...}, "S002": {...}, … }
    """
    with open(json_path, 'r') as f:
        subjects = json.load(f)
    return list(subjects.keys())


def iter_trial_paths_npy(patient_folder,
                        patient_id,
                        preprocessed_subfolder="preprocessed"):
    """
    Generates full paths to .npy files for a given patient:
    (day, block, trial, full_path)
    Uses the global variables days, blocks, trials.
    """
    for day, block, trial in product(days, blocks, trials):
        filename = f"{patient_id}_{day}_{block}_{trial}_preprocessed.npy"
        yield day, block, trial, os.path.join(patient_folder, preprocessed_subfolder, filename)


def get_all_npy_paths_by_group(subjects_dict, base_folders_map):
    """
    For each group in subjects_dict (e.g., "G01": [id1, id2, …]),
    constructs patient_folder = base_folders_map[group] + patient_id,
    iterates through trial .npy paths, and accumulates only the existing ones.
    """
    paths = []
    for group_code, subject_list in subjects_dict.items():
        base_folder = base_folders_map[group_code]
        for patient_id in subject_list:
            patient_folder = os.path.join(base_folder, patient_id)
            for day, block, trial, full_path in iter_trial_paths_npy(patient_folder, patient_id):
                if os.path.exists(full_path):
                    paths.append(full_path)
                else:
                    print(f"Warning: missing file {full_path}")
    return paths
