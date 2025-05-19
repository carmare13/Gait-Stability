import pandas as pd
from gait_events import gait_events_HC_JA

def segment_cycles(df):
    """
    Slice a DataFrame into gait cycles based on successive right-heel strikes.

    Args:
      df (pd.DataFrame): time-series of one trial (can be raw or downsampled).
    Returns:
      List[pd.DataFrame]: each DataFrame is one cycle (hs_R[i] → hs_R[i+1]).
    """
    # detect heel‐strike indices for the right foot
    hs_R, _, _, _ = gait_events_HC_JA(df)
    if len(hs_R) < 2:
        return []

    cycles = []
    for i in range(len(hs_R)-1):
        start, end = hs_R[i], hs_R[i+1]
        cycle = df.iloc[start:end].reset_index(drop=True)
        cycles.append(cycle)
    return cycles