import pandas as pd
import numpy as np
from scipy.signal import find_peaks 


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
    print("Heel Strikes (Right):", heel_strike_R)
    print("Heel Strikes (Left):", heel_strike_L) 
    # Return detected gait events
    return heel_strike_R, heel_strike_L, toe_off_R, toe_off_L   
     

def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def gait_events_norm(df):
    """
    Detects Heel Strikes (HS) and Toe-Offs (TO) using knee and hip angles and contact information.
    """

    # Extract required joint angles
    knee_flex_R = normalize_signal(df["Knee Flexion RT (deg)"].values)
    knee_flex_L = normalize_signal(df["Knee Flexion LT (deg)"].values)
    hip_flex_R = normalize_signal(df["Hip Flexion RT (deg)"].values)
    hip_flex_L = normalize_signal(df["Hip Flexion LT (deg)"].values)
    time = df["time"].values
    
    # Extract contact information (Contact LT and Contact RT)
    contact_R = df["Contact RT"].values  # Right foot contact
    contact_L = df["Contact LT"].values  # Left foot contact

    # Detect Heel Strikes (HS) - Knee extension peaks & hip flexion maximum
    hip_flex_threshold = 10  # Degrees (hip flexion threshold for validating HS)

    # Detect Heel Strikes (HS) - Knee extension peaks
    heel_strike_R, _ = find_peaks(knee_flex_R, height=0)  # Right knee extension peak
    heel_strike_L, _ = find_peaks(knee_flex_L, height=0)  # Left knee extension peak

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


def gait_events_simple(df):
    """
    Detecta Heel Strikes (HS) y Toe‐Offs (TO) basándose únicamente en las transiciones
    de los canales de contacto ("Contact RT" y "Contact LT").

    Asume que en df["Contact RT"] y df["Contact LT"]:
      - 0  = pie en el aire
      - 1000 = pie en apoyo

    Retorna cuatro listas de índices (enteros):
      hs_R: índices donde ocurre Heel‐Strike derecho
      hs_L: índices donde ocurre Heel‐Strike izquierdo
      to_R: índices donde ocurre Toe‐Off derecho
      to_L: índices donde ocurre Toe‐Off izquierdo
    """
    
    contact_R = df["Contact RT"].values > 0
    contact_L = df["Contact LT"].values > 0

    heel_strike_R = np.where((~contact_R[:-1]) & (contact_R[1:]))[0] + 1
    heel_strike_L = np.where((~contact_L[:-1]) & (contact_L[1:]))[0] + 1

    toe_off_R = np.where((contact_R[:-1]) & (~contact_R[1:]))[0] + 1
    toe_off_L = np.where((contact_L[:-1]) & (~contact_L[1:]))[0] + 1
    #print("Heel Strikes (Right):", len(heel_strike_R))
    #print("Heel Strikes (Left):", len(heel_strike_L)) 

    return heel_strike_R.tolist(), heel_strike_L.tolist(), toe_off_R.tolist(), toe_off_L.tolist()