# This script simulates the core backend logic for an HRV analysis application.
# It performs the following steps:
# 1. Generates mock RR-interval data to simulate input from a wearable device.
# 2. Calculates key HRV parameters from the mock data.
# 3. Plots and displays the analysis results for a single reading.
#
# This code is designed to be the "brain" of your backend, which would be
# triggered by an API request from the frontend.

import numpy as np
from scipy import signal
from collections import deque
import matplotlib.pyplot as plt
import os

def load_rr_intervals_from_file(filepath):
    """
    Simulates loading real RR-interval data from a file.
    In a production application, this data would come from a secure API call
    to a wearable device platform.

    Args:
        filepath (str): The path to the file containing RR-interval data.
        
    Returns:
        np.array or None: A NumPy array of RR-interval times in milliseconds,
                           or None if the file is not found or is empty.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    
    try:
        # We assume the file is a simple CSV with one column of numbers.
        rr_intervals = np.loadtxt(filepath)
        if len(rr_intervals) == 0:
            print("Warning: The data file is empty.")
            return None
        return rr_intervals
    except Exception as e:
        print(f"Error loading data from file: {e}")
        return None

def generate_mock_rr_intervals(n_beats=300):
    """
    Simulates a series of RR intervals from a wearable device.
    Real-world data would be retrieved from a device API.

    Args:
        n_beats (int): The number of heartbeats to simulate.

    Returns:
        np.array: A NumPy array of simulated RR-interval times in milliseconds.
    """
    # Simulate a resting heart rate around 65-75 bpm
    mean_rr = 60000 / np.random.uniform(65, 75)
    # Add random noise to simulate natural heart rate variability
    rr_intervals = np.random.normal(loc=mean_rr, scale=25, size=n_beats)
    return np.clip(rr_intervals, 500, 1200) # Ensure values are within a reasonable range

def calculate_hrv_parameters(rr_intervals):
    """
    Calculates key time-domain, frequency-domain, and non-linear HRV parameters.
    This function replaces the need for external libraries like pyHRV for this example.
    A production application would use dedicated libraries for robust, validated
    calculations.

    Args:
        rr_intervals (np.array): Cleaned RR-interval data in milliseconds.

    Returns:
        dict: A dictionary of calculated HRV parameters.
    """
    if len(rr_intervals) < 30:
        return {"error": "Not enough data for a meaningful analysis."}

    # Time-Domain Parameters
    diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diffs**2))
    sdnn = np.std(rr_intervals, ddof=1)
    pnn50 = (np.sum(np.abs(diffs) > 50) / len(diffs)) * 100

    # Frequency-Domain Parameters (using Lomb-Scargle for unevenly sampled data)
    # The following is a simplified example. For production, use a dedicated
    # HRV library's frequency-domain analysis function.
    # We will use Welch's method for demonstration, assuming 256ms sample rate for simplicity
    fs = 1000 / np.mean(rr_intervals)
    nperseg = min(256, len(rr_intervals))
    freqs, pxx = signal.welch(x=rr_intervals, fs=fs, nperseg=nperseg)
    
    # Define frequency bands
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    # Calculate power within each band
    lf_power = np.trapz(pxx[(freqs >= lf_band[0]) & (freqs < lf_band[1])], freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
    hf_power = np.trapz(pxx[(freqs >= hf_band[0]) & (freqs < hf_band[1])], freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
    
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0

    # Non-Linear Parameters (Poincaré Plot)
    # SD1 is the standard deviation of the distances of each point to the y=x line
    # SD2 is the standard deviation of the distances of each point to the y=-x+2*mean line
    sd1 = np.std(np.diff(rr_intervals) / np.sqrt(2))
    sd2 = np.std((rr_intervals[:-1] + rr_intervals[1:]) / np.sqrt(2))

    return {
        "rmssd": float(rmssd),
        "sdnn": float(sdnn),
        "pnn50": float(pnn50),
        "lf_power": float(lf_power),
        "hf_power": float(hf_power),
        "lf_hf_ratio": float(lf_hf_ratio),
        "sd1": float(sd1),
        "sd2": float(sd2),
    }

def plot_hrv_analysis(rr_intervals, freqs, pxx, sd1, sd2):
    """
    Generates and displays plots for HRV analysis.

    Args:
        rr_intervals (np.array): The RR-interval data.
        freqs (np.array): Frequencies from the power spectral density.
        pxx (np.array): Power spectral density.
        sd1 (float): SD1 value for the Poincaré plot.
        sd2 (float): SD2 value for the Poincaré plot.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('HRV Analysis Plots', fontsize=16)

    # Plot 1: Time-Domain Tachogram
    ax1.plot(rr_intervals, color='b', label='RR Intervals')
    ax1.set_title('Time-Domain Tachogram')
    ax1.set_xlabel('Beat Number')
    ax1.set_ylabel('RR Interval (ms)')
    ax1.grid(True)

    # Plot 2: Frequency-Domain Power Spectrum
    ax2.plot(freqs, pxx, color='g', label='Power Spectral Density')
    ax2.set_title('Frequency-Domain Power Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power ($ms^2/Hz$)')
    ax2.set_xlim(0, 0.5) # Focus on the relevant HRV frequency bands
    ax2.grid(True)

    # Plot 3: Poincaré Plot
    rr_x = rr_intervals[:-1]
    rr_y = rr_intervals[1:]
    ax3.scatter(rr_x, rr_y, alpha=0.5, color='r')
    ax3.set_title('Poincaré Plot')
    ax3.set_xlabel('$RR_n (ms)$')
    ax3.set_ylabel('$RR_{n+1} (ms)$')
    ax3.axvline(x=np.mean(rr_intervals), color='k', linestyle='--', linewidth=1)
    ax3.axhline(y=np.mean(rr_intervals), color='k', linestyle='--', linewidth=1)
    ax3.text(1.05, 0.5, f'SD1: {sd1:.2f} ms\nSD2: {sd2:.2f} ms',
             transform=ax3.transAxes, ha='left', va='center')
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    """
    The main execution block of the script.
    """
    print("Starting HRV analysis simulation...")
    
    # Option 1: Load data from a file (simulating real data)
    file_path = 'rr_intervals.csv'
    rr_intervals = load_rr_intervals_from_file(file_path)

    # Option 2: Generate mock data if file loading fails
    if rr_intervals is None:
        print("Using mock data for demonstration purposes.")
        rr_intervals = generate_mock_rr_intervals()
    
    # Step 2: Perform HRV analysis
    analysis_results = calculate_hrv_parameters(rr_intervals)
    
    # Step 3: Display results
    if "error" in analysis_results:
        print(f"Error: {analysis_results['error']}")
    else:
        print("\n--- HRV Analysis Results ---")
        print(f"RMSSD: {analysis_results['rmssd']:.2f} ms")
        print(f"SDNN: {analysis_results['sdnn']:.2f} ms")
        print(f"pNN50: {analysis_results['pnn50']:.2f} %")
        print(f"LF Power: {analysis_results['lf_power']:.2f} ms^2")
        print(f"HF Power: {analysis_results['hf_power']:.2f} ms^2")
        print(f"LF/HF Ratio: {analysis_results['lf_hf_ratio']:.2f}")
        print(f"Poincaré Plot (SD1): {analysis_results['sd1']:.2f} ms")
        print(f"Poincaré Plot (SD2): {analysis_results['sd2']:.2f} ms")
        print("\n--- End of Analysis ---")

    # Step 4: Plot the results
    # To plot the frequency and Poincare plots, we need the initial data
    # that was generated in calculate_hrv_parameters. We will make a change
    # to the function to get those values out of the function as a returned
    # value. 
    # For now, we will re-calculate the values here as a demonstration.
    
    fs = 1000 / np.mean(rr_intervals)
    nperseg = min(256, len(rr_intervals))
    freqs, pxx = signal.welch(x=rr_intervals, fs=fs, nperseg=nperseg)
    
    diffs = np.diff(rr_intervals)
    sd1 = np.std(np.diff(rr_intervals) / np.sqrt(2))
    sd2 = np.std((rr_intervals[:-1] + rr_intervals[1:]) / np.sqrt(2))
    
    plot_hrv_analysis(rr_intervals, freqs, pxx, sd1, sd2)

if __name__ == "__main__":
    main()
