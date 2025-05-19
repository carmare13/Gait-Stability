import os
import pandas as pd
from tqdm import tqdm
from Data_loader import load_patient_data

def ensure_dir(path):
    """Create directory if it doesn’t exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def save_trial_summary(df_summary, output_folder, patient_id, day, block, trial):
    """
    Save the summary DataFrame for one trial to CSV.
    File named: {patient_id}_{day}_{block}_{trial}_summary.csv
    """
    ensure_dir(output_folder)
    filename = f"{patient_id}_{day}_{block}_{trial}_summary.csv"
    out_path = os.path.join(output_folder, filename)
    df_summary.to_csv(out_path, index=False)

def streaming_summary(group_folders, group_codes, output_path):
    """
    For each group in `group_folders` (with matching `group_codes`), compute:
      - group summary (n_patients, total_expected_trials, total_loaded_trials)
      - patient summary (group_code, patient_id, n_trials_expected, n_trials_loaded, n_trials_with_nans)
    and write them to an Excel file at `output_path` with two sheets:
      • Group_Summary
      • Patient_Summary
    """
    patient_rows = []
    group_rows   = []

    # list of your days/blocks/trials; adjust if you have them elsewhere
    days   = ["D01","D02"]
    blocks = ["B01","B02","B03"]
    trials = ["T01","T02","T03"]
    expected_per_patient = len(days)*len(blocks)*len(trials)

    # variables to check for NaNs
    ivars = [
        'Ankle Dorsiflexion RT (deg)', 'Ankle Dorsiflexion LT (deg)',
        'Noraxon MyoMotion-Trajectories-Heel LT-x (mm)',
        'Noraxon MyoMotion-Trajectories-Heel LT-y (mm)',
        'Noraxon MyoMotion-Trajectories-Heel RT-y (mm)',
        'Noraxon MyoMotion-Trajectories-Heel RT-x (mm)',
        'Knee Flexion RT (deg)', 'Knee Flexion LT (deg)',
        'Hip Flexion RT (deg)', 'Hip Flexion LT (deg)',
    ]

    for base_folder, grp in zip(group_folders, group_codes):
        patient_ids = sorted([
            d for d in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, d)) and d.startswith("S")
        ])
        total_expected = 0
        total_loaded   = 0

        for pid in tqdm(
            patient_ids,
            desc=f"Patients {grp}",
            leave=False
        ):
            # load data (memory‐safe)
            df = load_patient_data(
                os.path.join(base_folder, pid),
                pid, grp, verbose=False
            )

            # count loaded trials
            trials_df = df[['day','block','trial']].drop_duplicates()
            n_loaded  = len(trials_df)
            total_expected += expected_per_patient
            total_loaded   += n_loaded

            # count trials with any NaN
            n_nans = 0
            for _, sub in trials_df.iterrows():
                sel = df[
                    (df['day']==sub['day']) &
                    (df['block']==sub['block']) &
                    (df['trial']==sub['trial'])
                ]
                if sel[ivars].isna().any().any():
                    n_nans += 1

            patient_rows.append({
                'group_code':         grp,
                'patient_id':         pid,
                'n_trials_expected':  expected_per_patient,
                'n_trials_loaded':    n_loaded,
                'n_trials_with_nans': n_nans
            })

        group_rows.append({
            'group_code':            grp,
            'n_patients':            len(patient_ids),
            'total_expected_trials': total_expected,
            'total_loaded_trials':   total_loaded
        })

    # build small DataFrames
    df_grp = pd.DataFrame(group_rows)
    df_pat = pd.DataFrame(patient_rows)

    # write to Excel
    ensure_dir(os.path.dirname(output_path))
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df_grp.to_excel(writer, sheet_name='Group_Summary',   index=False)
        df_pat.to_excel(writer, sheet_name='Patient_Summary', index=False)