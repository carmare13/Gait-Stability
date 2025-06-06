import os
import pandas as pd
from itertools import product

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


def load_patient_data(patient_folder, patient_id, group_code, subfolder=None, verbose=False):
    """
    Read all available trial CSVs for one patient into a single DataFrame.
    Adds metadata columns: patient_id, group, day, block, trial.
    Returns:
        dfs (list of DataFrames), paths (list of str)
    """
    df_list = []
    file_list = []
    
    for d, b, t, path in iter_trial_paths(patient_folder, patient_id, group_code):
        #print(f"Accediendo a: {path}")
        if not os.path.isfile(path):
            if verbose:
                print(f"  [WARN] file not found: {path}")
            continue
        
        try:
            df = pd.read_csv(path, dtype=float)
            #print(f"Trying to read{path}: shape={df.shape}")
            if df.empty:
                print(f"[WARN] empty file: {path}")
                continue
            # Attach metadata...
            df['patient_id'] = patient_id
            df['group']      = group_code
            df['day']        = d
            df['block']      = b
            df['trial']      = t
            df_list.append(df)
            file_list.append(path)
        except Exception as e:
            print(f"[ERROR] reading {path}: {e}")

    
    if df_list:
        return df_list, file_list
    else:
        return [], [] # No data for this patient 



def summarize_file(path, usecols=None, dtype=None):
    """
    Read a single CSV at `path`, compute descriptive stats and count missing values.
    Returns a DataFrame with the transpose of describe() and a 'n_missing' column.
    """
    df = pd.read_csv(path, usecols=usecols, dtype=dtype)
    summary = df.describe().transpose()
    summary['n_missing'] = df.isna().sum()
    return summary