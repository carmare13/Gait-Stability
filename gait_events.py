import pandas as pd
import numpy as np
from scipy.signal import find_peaks  # Importar find_peaks para detectar picos

def gait_events_HC_JA(df):
    """
    Detects Heel Strikes (HS) and Toe-Offs (TO) using knee and hip angles and contact information.
    """

    # Extract required joint angles
    knee_flex_R = df["Knee Flexion RT (deg)"].values
    knee_flex_L = df["Knee Flexion LT (deg)"].values
    hip_flex_R = df["Hip Flexion RT (deg)"].values
    hip_flex_L = df["Hip Flexion LT (deg)"].values
    time = df["time"].values
    
    # Extract contact information (Contact LT and Contact RT)
    contact_R = df["Contact RT"].values  # Right foot contact
    contact_L = df["Contact LT"].values  # Left foot contact

    # Detect Heel Strikes (HS) - Knee extension peaks & hip flexion maximum
    hip_flex_threshold = 10  # Degrees (hip flexion threshold for validating HS)

    # Detect Heel Strikes (HS) - Knee extension peaks
    heel_strike_R, _ = find_peaks(knee_flex_R, height=0)  # Right knee extension peak
    heel_strike_L, _ = find_peaks(knee_flex_L, height=0)  # Left knee extension peak

    # Filter by hip flexion (only accept if hip is flexed beyond threshold)
    #heel_strike_R = [i for i in heel_strike_R if hip_flex_R[i] > hip_flex_threshold]
    #heel_strike_L = [i for i in heel_strike_L if hip_flex_L[i] > hip_flex_threshold]

    # Filter by Contact information (only consider if contact is made - contact == 1000)
    heel_strike_R = [i for i in heel_strike_R if contact_R[i] == 1000]
    heel_strike_L = [i for i in heel_strike_L if contact_L[i] == 1000]

    # Toe-Off Detection - Knee flexion velocity and hip acceleration peaks
    knee_vel_R = np.gradient(knee_flex_R, time)  # Right knee velocity
    knee_vel_L = np.gradient(knee_flex_L, time)  # Left knee velocity

    # Compute Hip Acceleration (Second Derivative of Hip Flexion)
    hip_vel_R = np.gradient(hip_flex_R, time)  # Right hip velocity
    hip_vel_L = np.gradient(hip_flex_L, time)  # Left hip velocity
    hip_acc_R = np.gradient(hip_vel_R, time)  # Right hip acceleration
    hip_acc_L = np.gradient(hip_vel_L, time)  # Left hip acceleration

    # Detect peaks in knee velocity (possible toe-offs)
    vel_peaks_R, _ = find_peaks(knee_vel_R, distance=30)  # Right knee velocity peaks
    vel_peaks_L, _ = find_peaks(knee_vel_L, distance=30)  # Left knee velocity peaks

    # Select the highest knee velocity peak within each gait cycle and validate with hip acceleration
    toe_off_R = []
    toe_off_L = []

    for i in range(len(heel_strike_R) - 1):
        cycle_peaks = [p for p in vel_peaks_R if heel_strike_R[i] < p < heel_strike_R[i + 1]]
        if cycle_peaks:
            max_peak = max(cycle_peaks, key=lambda p: knee_vel_R[p])
            # Validate Toe-Off with Hip Acceleration (must be positive, indicating forward motion)
            if hip_acc_R[max_peak] > 0:
                toe_off_R.append(max_peak)

    for i in range(len(heel_strike_L) - 1):
        cycle_peaks = [p for p in vel_peaks_L if heel_strike_L[i] < p < heel_strike_L[i + 1]]
        if cycle_peaks:
            max_peak = max(cycle_peaks, key=lambda p: knee_vel_L[p])
            if hip_acc_L[max_peak] > 0:
                toe_off_L.append(max_peak)

    # Return detected gait events
    return heel_strike_R, heel_strike_L, toe_off_R, toe_off_L     








      

    