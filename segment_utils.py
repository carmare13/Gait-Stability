import pandas as pd
from gait_events import gait_events_HC_JA, gait_events_norm, gait_events_simple

def segment_cycles(df,print_cycle_length=False):
    """
    Slice a DataFrame into gait cycles based on successive right-heel strikes.
    Uses a queue to store the cycles

    Args:
      df (pd.DataFrame): time-series of one trial (can be raw or downsampled).
      print_cycle_length (bool): If True, prints the length of each cycle.
    Returns:
      Deque[pd.DataFrame]: each DataFrame is one cycle (hs_R[i] → hs_R[i+1]).
    """
    # detect heel‐strike indices for the right foot
    hs_R, _, _, _ = gait_events_HC_JA(df)
    if len(hs_R) < 2:
        return []  # No cycles to segment

    cycles = []
    for i in range(len(hs_R)-1):
        start, end = hs_R[i], hs_R[i+1]
        cycle = df.iloc[start:end].reset_index(drop=True)
        if print_cycle_length:
            print(f"Cycle {i}: length = {end - start} samples")
        cycles.append(cycle)
    return cycles

def segment_cycles_norm(df,print_cycle_length=False):
    """
    Slice a DataFrame into gait cycles based on successive right-heel strikes.
    Uses a queue to store the cycles
    Works with normalized signal for the segmentation.

    Args:
      df (pd.DataFrame): time-series of one trial (can be raw or downsampled).
      print_cycle_length (bool): If True, prints the length of each cycle.
    Returns:
      Deque[pd.DataFrame]: each DataFrame is one cycle (hs_R[i] → hs_R[i+1]).
    """
    # detect heel‐strike indices for the right foot
    hs_R, _, _, _ = gait_events_norm(df)
    if len(hs_R) < 2:
        return []  # No cycles to segment

    cycles = []
    for i in range(len(hs_R)-1):
        start, end = hs_R[i], hs_R[i+1]
        cycle = df.iloc[start:end].reset_index(drop=True)
        if print_cycle_length:
            print(f"Cycle {i}: length = {end - start} samples")
        cycles.append(cycle)
    return cycles

def segment_cycles_simple(df,print_cycle_length=False):
    """
    Slice a DataFrame into gait cycles based on successive right-heel strikes.
    One gait cycle is defined as the time between two successive right-heel strikes.
    Uses a queue to store the cycles

    Args:
      df (pd.DataFrame): time-series of one trial (can be raw or downsampled).
      print_cycle_length (bool): If True, prints the length of each cycle.
    Returns:
      Deque[pd.DataFrame]: each DataFrame is one cycle (hs_R[i] → hs_R[i+1]).
    """
    # detect heel‐strike indices for the right foot
    hs_R, _, _, _ = gait_events_simple(df)
    if len(hs_R) < 2:
        return []  # No cycles to segment

    cycles = []
    for i in range(len(hs_R)-1):
        start, end = hs_R[i], hs_R[i+1]
        cycle = df.iloc[start:end].reset_index(drop=True)
        if print_cycle_length:
            print(f"Cycle {i}: length = {end - start} samples")
        cycles.append(cycle)
    return cycles