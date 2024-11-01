# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + endofcell="--"
import sys
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.colors as colors
import openpyxl
import plotnine
import numpy
import seaborn as sns
from plotnine import *
import neurokit2 as nk
#import patchworklib as pw
from scipy import signal, datasets
import pandas as pd

import math
# -

import warnings
warnings.filterwarnings('ignore')

# move to src directory from the current directory to import QTC_nomogram
current_directory = os.getcwd()
src_directory = os.path.join(current_directory, '..', '..', 'src')
sys.path.append(src_directory)
# --

# +
# Path to the directory containing utils.py
utils_path = os.path.abspath('../preprocess_european_st')

# Add the path to sys.path
sys.path.insert(0, utils_path)

# Import the utils module
import utils


# +
# Define the sampling rate
sampling_rate = 1000  # samples per second

# Define the number of samples
num_samples = 38401  # Adjust this according to your requirement

# Calculate the time interval
time_interval = 1 / sampling_rate

# Generate the time column
time_column = [i * time_interval for i in range(num_samples)]

# Convert the time column to a DataFrame
ECG_time = pd.DataFrame({'ms': time_column})

# Display the DataFrame
print(ECG_time)
# -

ECG_time.ms

#ECG_time = qn.read_time_file("../../data/time/time.csv")
mydir = "/Users/user/ECG-X_DataReader/data/csv_format"
myfiles = glob.glob(mydir + "/*.csv")
myfiles


# # Step 1: Read 12 lead

def read_12_lead_ecg_data(file_path, ECG_time, leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']):
    """
    Read ECG data from a CSV file and return it as a DataFrame.

    This function reads a CSV file containing ECG data, associates it with the provided
    ECG time data, and returns the combined data as a DataFrame with columns "time" and the specified 12 leads.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing ECG data.
    ECG_time : pandas.DataFrame
        A DataFrame containing ECG time data.

    Returns
    -------
    ECG : pandas.DataFrame
        A DataFrame containing the combined ECG data with columns "time" and the specified 12 leads.
    """
    
    # Read ECG data from CSV
    ecg_data = pd.read_csv(file_path)
    
    # Extract the specified 12 leads
    leads_data = ecg_data[leads]
    
    # Assign ECG time data
    ecg_time_column = ECG_time["ms"]  # Assuming the time column name in ECG_time is 'ms'
    leads_data['time'] = ecg_time_column
    
    return leads_data


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

def butterworth_filter(ECG_mv, ECG_time, sampling_rate, lowcut=0.5, highcut=50.0, order=1, plot_result=True):
    """
    Apply a Butterworth filter to an ECG signal.

    Args:
    - ECG_mv (pandas.DataFrame): DataFrame containing ECG signal data for all 12 leads.
    - ECG_time (numpy.ndarray): The corresponding time data for the ECG signal.
    - sampling_rate (float): Sampling rate of the ECG signal.
    - lowcut (float, optional): Lower cutoff frequency in Hz. Default is 0.5 Hz.
    - highcut (float, optional): Upper cutoff frequency in Hz. Default is 50.0 Hz.
    - order (int, optional): Filter order. Default is 1.
    - plot_result (bool, optional): If True, plot the original and filtered signals for each lead. Default is True.

    Returns:
    - ECG_mv (pandas.DataFrame): DataFrame containing the filtered ECG signal data for all 12 leads.
    """
    # Normalize the cutoff frequencies by the Nyquist frequency
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq

    # Design the Butterworth filter
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply the filter to each lead column
    for column in ECG_mv.columns:
        original_signal = ECG_mv[column].copy()  # Store the original signal
        ECG_mv[column] = signal.filtfilt(b, a, ECG_mv[column])
        if plot_result:
            # Plot the original and filtered signals
            fig, axs = plt.subplots(2, 1, figsize=(15, 10))
            axs[0].plot(ECG_time, original_signal)
            axs[1].plot(ECG_time, ECG_mv[column])
            axs[0].set_xlabel('Time (s)')
            axs[1].set_xlabel('Time (s)')
            axs[0].set_title('Original Noised Signal ' + column)
            axs[1].set_title('Denoised Signal ' + column)
            axs[0].set_ylim([-3, +3])
            axs[1].set_ylim([-3, +3])
            plt.tight_layout()
            plt.show()

    return ECG_mv



# -

# # Step 2: Correct Baseline Wandering

def correct_baseline_wandering(ECG_mv, ECG_time, order=4, cutoff_frequency=0.5, sampling_rate=360, plot_result=True):
    """
    Correct baseline wandering in ECG data for all 12 leads using a high-pass Butterworth filter.

    Parameters
    ----------
    ECG_mv : pandas.DataFrame
        DataFrame containing ECG signal data for all 12 leads.
    ECG_time : numpy.ndarray
        The corresponding time data for the ECG signal.
    order : int, optional
        The filter order for the Butterworth filter. Default is 4.
    cutoff_frequency : float, optional
        The cutoff frequency for the high-pass filter in Hz. Default is 0.5 Hz.
    sampling_rate : int, optional
        The sampling rate of the ECG signal. Default is 360 Hz.
    plot_result : bool, optional
        If True, plot the original and filtered signals for each lead. Default is True.

    Returns
    -------
    ECG_mv : pandas.DataFrame
        DataFrame containing the corrected ECG signal data for all 12 leads.

    """
    b, a = signal.butter(order, cutoff_frequency, btype='highpass', fs=sampling_rate)
    
    # Apply the high-pass filter to each lead column
    for column in ECG_mv.columns:
        original_signal = ECG_mv[column].copy()  # Store the original signal
        ECG_mv[column] = signal.filtfilt(b, a, ECG_mv[column])
        if plot_result:
            # Plot the original and filtered signals
            fig, axs = plt.subplots(2, 1, figsize=(15, 10))
            axs[0].plot(ECG_time, original_signal)
            axs[1].plot(ECG_time, ECG_mv[column])
            axs[0].set_xlabel('Time (s)')
            axs[1].set_xlabel('Time (s)')
            axs[0].set_title('Original Wandering Signal ' + column)
            axs[1].set_title('Butterworth Filtered Signal ' + column)
            axs[0].set_ylim([-3, +3])
            axs[1].set_ylim([-3, +3])
            plt.tight_layout()
            plt.show()
        
    return ECG_mv


# # Step 3: Invert signal

# +
import pandas as pd
import neurokit2 as nk

def read_and_invert_12_lead_ecg_data(ECG, ECG_time, leads=['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']):
    """
    Read ECG data, invert the signals, and return as a DataFrame.

    Parameters
    ----------
    ECG : pandas.DataFrame
        A DataFrame containing ECG data.
    ECG_time : pandas.DataFrame
        A DataFrame containing ECG time data.
    leads : list of str
        The list of lead names to extract and invert.

    Returns
    -------
    inverted_ECG : pandas.DataFrame
        A DataFrame containing the combined ECG data with columns "time" and the specified 12 leads
        with inverted signals.
    """

    # Check if all specified leads are in the DataFrame
    missing_leads = [lead for lead in leads if lead not in ECG.columns]
    if missing_leads:
        raise KeyError(f"The following leads are missing from the ECG data: {missing_leads}")

    # Extract the specified leads
    leads_data = ECG[leads]

    # Assign ECG time data
    leads_data['time'] = ECG_time["ms"]  # Assuming the time column name in ECG_time is 'ms'

    # Invert the signals
    inverted_ecg = leads_data.copy()
    for lead in leads:
        inverted_ecg[lead] = invert_signal(leads_data[lead])

    return inverted_ecg

# Define the invert_signal function
def invert_signal(ECG_mv):
    """
    Invert the ECG signal for peak detection.

    Parameters
    ----------
    ECG_mv : pandas.Series or ndarray
        The ECG signal to be inverted.

    Returns
    -------
    inverted_signal : ndarray
        The inverted ECG signal.
    """
    inverted_signal, is_inverted = nk.ecg_invert(ECG_mv, sampling_rate=200, show=False)
    return inverted_signal



# -

# # Step 4: Find QRS

def find_R_peaks(ECG, leads= ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']):
    """
    Find ECG peaks in the provided ECG data.

    This function identifies ECG peaks by analyzing the 'mv' column of the ECG DataFrame.

    Parameters
    ----------
    ECG : pandas.DataFrame
        A DataFrame containing ECG data with 'mv' column.

    Returns
    -------
    peaks : dict
        A dictionary containing R-peaks time for each lead.
    """
    peaks = {}

    for lead in leads:
        R_Peaks = nk.ecg_peaks(ECG[lead], sampling_rate=1000, method='martinez2004', correct_artifacts=False, show=False)
        peaks[lead] = R_Peaks[1].get('ECG_R_Peaks')

    return peaks


# +
import matplotlib.pyplot as plt

def plot_12_lead_ecg(ECG_mv, ECG_time, fig_title, start=0, end=10000, figsize=(20, 35)):
    """
    Plot 12-lead ECG signals on one figure.

    Parameters
    ----------
    ECG_mv : pandas.DataFrame
        DataFrame containing ECG signal data for all 12 leads.
    ECG_time : numpy.ndarray
        The corresponding time data for the ECG signal.
    fig_title : str
        The title for the figure.
    start : int, optional
        The start index for the data to plot. Default is 0.
    end : int, optional
        The end index for the data to plot. Default is 10000.
    figsize : tuple, optional
        Figure size for the plot. Default is (20, 35).
    """
    # Drop the last column
    ECG_mv = ECG_mv.iloc[:, :-1]
    
    leads = ECG_mv.columns
    num_leads = len(leads)
    
    fig, axs = plt.subplots(num_leads, 1, figsize=figsize, sharex=True)
    fig.suptitle(fig_title, fontsize=16)

    for i, lead in enumerate(leads):
        axs[i].plot(ECG_time[start:end], ECG_mv[lead][start:end])
        axs[i].set_title(lead)
        axs[i].set_ylabel('mV')
        axs[i].grid(True)
        
    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.savefig('../../results/ptb-diagnostic-ecg-database-1.0.0/denoised_signal/' + str(fig_title) + '.png')
    plt.show()



# + active=""
#
# for f in range(len(myfiles)):
#     print("\n\n\n-----------------" + str(os.path.basename(myfiles[f])) + "-----------------\n")
#     # Read ECG data
#     ECG = read_12_lead_ecg_data(myfiles[f], ECG_time)
#     # read just first 10000 msecs, 10 secs lead
#     ECG_mv = correct_baseline_wandering(ECG[0:10000], ECG_time[0:10000], plot_result=False)
#     # denoise signal # Apply Butterworth filter to ECG signals
#     filtered_ecgs = butterworth_filter(ECG_mv[0:10000],ECG_time[0:10000], 1000, plot_result=False)
#     fig_title = str(os.path.basename(myfiles[f]))
#     plot_12_lead_ecg(filtered_ecgs, ECG_time, fig_title)
#     #plot_12_lead_ecg(ECG[0:10000], ECG_time[0:10000], fig_title)

# +
import numpy as np
import matplotlib.pyplot as plt

def plot_ecg_with_peaks(ecg_signal, r_peak_indices_dict, j_points_dict, fig_title):
    for lead in r_peak_indices_dict:
        plt.figure(figsize=(12, 6))
        r_peaks = r_peak_indices_dict[lead]
        j_points = j_points_dict[lead]

        plt.plot(ecg_signal[lead], label=f'ECG Lead {lead}')
        plt.scatter(r_peaks, ecg_signal[lead].iloc[r_peaks], color='red', label='R-peaks')
        plt.scatter(j_points, ecg_signal[lead].iloc[j_points], color='blue', label='J-points')
        
        plt.title(f'ECG Lead {lead}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.savefig(f'../../results/ptb-diagnostic-ecg-database-1.0.0/j-point/slope/{fig_title}.png')
        plt.show()



# -


# # 1. Threshold-Based Window Search J-Point Detection Algorithm

# +
import numpy as np

def detect_j_point(ecg_signal, r_peak_indices_dict, threshold=0.5):
    j_points_dict = {}
    for lead, r_peak_indices in r_peak_indices_dict.items():
        j_points = []
        for r_peak in r_peak_indices:
            # Extract signal data as a NumPy array
            ecg_data = ecg_signal[lead].values if isinstance(ecg_signal, pd.DataFrame) else ecg_signal
            # Define a window to search for the J-point after the R-peak
            search_window = ecg_data[r_peak:r_peak+30]  # 30 samples after the R-peak
            # Find the point where the signal drops below the threshold
            for i in range(len(search_window)):
                if search_window[i] < threshold:
                    j_points.append(r_peak + i)
                    break
        j_points_dict[lead] = j_points
    return j_points_dict



# -

# # 1.1. Acceptance Testng

# + active=""
# for f in range(len(myfiles)):
#     
#     print("\n\n\n-----------------" + str(os.path.basename(myfiles[f])) + "-----------------\n")
#
#     # Read ECG data
#     ECG = read_12_lead_ecg_data(myfiles[f], ECG_time, leads = ['i', 'ii'])
#     # read just first 10000 msecs, 10 secs lead
#     ECG_mv = correct_baseline_wandering(ECG[0:10000], ECG_time[0:10000], plot_result=False)
#     # denoise signal # Apply Butterworth filter to ECG signals
#     filtered_ecgs = butterworth_filter(ECG_mv[0:10000],ECG_time[0:10000], 1000, plot_result=False)
#     # invert signal where required
#     inverted_ECG = read_and_invert_12_lead_ecg_data(filtered_ecgs[0:3000], ECG_time[0:3000], leads = ['i', 'ii'])
#     print(type(inverted_ECG))
#     # Find R peaks
#     R_peaks_time = find_R_peaks(inverted_ECG, leads = ['i', 'ii'])
#     
#     r_peak_indices = R_peaks_time
#     #r_peak_indices = list(map(int, R_peaks_time))
#     # jpoint detection
#     j_points = detect_j_point(inverted_ECG, r_peak_indices)(inverted_ECG, r_peak_indices)
#
#     # Plot the ECG signal and detected J-points
#     # Assuming inverted_ECG is your DataFrame and r_peak_indices and j_points are your dictionaries
#     plot_ecg_with_peaks(inverted_ECG, r_peak_indices, j_points)
# -

# # 2. Slope-Based J-Point Detection Algorithm

# +
# 2
import numpy as np
import pandas as pd

def detect_j_point_slope(ecg_signal, r_peak_indices_dict, window_sizes):
    best_window_size = None
    best_accuracy = 0
    j_points_dict = {}
    
    for window_size in window_sizes:
        temp_j_points_dict = {}
        for lead, r_peak_indices in r_peak_indices_dict.items():
            j_points = []
            ecg_data = ecg_signal[lead].values if isinstance(ecg_signal, pd.DataFrame) else ecg_signal
            for r_peak in r_peak_indices:
                search_window = ecg_data[r_peak:r_peak + window_size]
                derivative = np.diff(search_window)
                min_derivative_index = np.argmin(derivative)
                j_point = r_peak + min_derivative_index + 1
                j_points.append(j_point)
            temp_j_points_dict[lead] = j_points
        
        # Validate this window size (implement your validation logic here)
        accuracy = validate_j_points(temp_j_points_dict)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_window_size = window_size
            j_points_dict = temp_j_points_dict
    
    print(f"Best window size: {best_window_size} with accuracy: {best_accuracy}")
    return j_points_dict

def validate_j_points(j_points_dict):
    # Implement your validation logic to compare detected J-points with annotated data
    # Return an accuracy metric (e.g., percentage of correctly detected J-points)
    return np.random.rand()  # Placeholder for demonstration





# + active=""
# for f in range(len(myfiles)):
#     
#     print("\n\n\n-----------------" + str(os.path.basename(myfiles[f])) + "-----------------\n")
#
#     # Read ECG data
#     ECG = read_12_lead_ecg_data(myfiles[f], ECG_time, leads = ['ii'])#, 'ii'])
#     # read just first 10000 msecs, 10 secs lead
#     ECG_mv = correct_baseline_wandering(ECG[0:10000], ECG_time[0:10000], plot_result=False)
#     # denoise signal # Apply Butterworth filter to ECG signals
#     filtered_ecgs = butterworth_filter(ECG_mv[0:10000],ECG_time[0:10000], 1000, plot_result=False)
#     # invert signal where required
#     inverted_ECG = read_and_invert_12_lead_ecg_data(filtered_ecgs[0:10000], ECG_time[0:10000], leads = ['ii'])#, 'ii'])
#     print(type(inverted_ECG))
#     # Find R peaks
#     R_peaks_time = find_R_peaks(inverted_ECG, leads = ['ii'])#, 'ii'])
#     
#     r_peak_indices = R_peaks_time
#     #r_peak_indices = list(map(int, R_peaks_time))
#     # Test window sizes from 20 to 60 samples
#     window_sizes = range(20, 60)
#     j_points = detect_j_point_slope(inverted_ECG, r_peak_indices, window_sizes)
#     fig_title = str(os.path.basename(myfiles[f]))
#     # Plot the ECG signal and detected J-points
#     # Assuming inverted_ECG is your DataFrame and r_peak_indices and j_points are your dictionaries
#     plot_ecg_with_peaks(inverted_ECG, r_peak_indices, j_points, fig_title)
#     
#     print(j_points)
# +
import numpy as np
import pandas as pd

def detect_j_point_slope(ecg_signal, r_peak_indices_dict, window_size=30):
    j_points_dict = {}
    for lead, r_peak_indices in r_peak_indices_dict.items():
        j_points = []
        ecg_data = ecg_signal[lead].values if isinstance(ecg_signal, pd.DataFrame) else ecg_signal
        for r_peak in r_peak_indices:
            search_window = ecg_data[r_peak:r_peak + window_size]
            derivative = np.diff(search_window)
            min_derivative_index = np.argmin(derivative)
            j_point = r_peak + min_derivative_index + 1
            j_points.append(j_point)
        j_points_dict[lead] = j_points
    return j_points_dict


# + active=""
# for f in range(len(myfiles)):
#     
#     print("\n\n\n-----------------" + str(os.path.basename(myfiles[f])) + "-----------------\n")
#
#     # Read ECG data
#     ECG = read_12_lead_ecg_data(myfiles[f], ECG_time, leads = ['i'])#, 'ii'])
#     # read just first 10000 msecs, 10 secs lead
#     ECG_mv = correct_baseline_wandering(ECG[0:10000], ECG_time[0:10000], plot_result=False)
#     # denoise signal # Apply Butterworth filter to ECG signals
#     filtered_ecgs = butterworth_filter(ECG_mv[0:10000],ECG_time[0:10000], 1000, plot_result=False)
#     # invert signal where required
#     inverted_ECG = read_and_invert_12_lead_ecg_data(filtered_ecgs[0:10000], ECG_time[0:10000], leads = ['i'])#, 'ii'])
#     print(type(inverted_ECG))
#     # Find R peaks
#     R_peaks_time = find_R_peaks(inverted_ECG, leads = ['i'])#, 'ii'])
#     
#     r_peak_indices = R_peaks_time
#     #r_peak_indices = list(map(int, R_peaks_time))
#     # jpoint detection
#     j_points = detect_j_point_slope(inverted_ECG, r_peak_indices)
#     fig_title = str(os.path.basename(myfiles[f]))
#     # Plot the ECG signal and detected J-points
#     # Assuming inverted_ECG is your DataFrame and r_peak_indices and j_points are your dictionaries
#     plot_ecg_with_peaks(inverted_ECG, r_peak_indices, j_points, fig_title)
# -

def plot_ecg_with_peaks(ecg_signals, r_peak_indices, r_offsets, sub_folder, fig_title):
    plt.figure(figsize=(12, 10))
    
    for i, (lead, signal) in enumerate(ecg_signals.items(), start=1):
        plt.subplot(2, 1, i)  # Create a subplot for each lead
        plt.plot(signal, label=f'ECG Lead {lead}')
        
        # Filter valid R-peaks
        valid_r_peaks = [r_peak for r_peak in r_peak_indices[lead] if 0 <= r_peak < len(signal)]
        plt.scatter(valid_r_peaks, signal.iloc[valid_r_peaks], color='red', label=f'R-peaks {lead}')
        
        # Filter valid R-offsets
        valid_r_offsets = [r_offset for r_offset in r_offsets[lead] if 0 <= r_offset < len(signal)]
        plt.scatter(valid_r_offsets, signal.iloc[valid_r_offsets], color='blue', label=f'R-offsets {lead}')
        
        plt.title(f'ECG Signal with R-peaks and R-offsets for Lead {lead}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'../../results/ptb-diagnostic-ecg-database-1.0.0/j-point/{sub_folder}/{fig_title}.png')
    plt.show()


def plot_ecg_with_peaks(ecg_signals, r_peak_indices, r_offsets, sub_folder, fig_title):
    plt.figure(figsize=(12, 10))
    
    for i, (lead, signal) in enumerate(ecg_signals.items(), start=1):
        plt.subplot(2, 1, i)  # Create a subplot for each lead
        plt.plot(signal, label=f'ECG Lead {lead}')
        
        # Filter valid R-peaks
        valid_r_peaks = [r_peak for r_peak in r_peak_indices[lead] if 0 <= r_peak < len(signal)]
        plt.scatter(valid_r_peaks, signal.iloc[valid_r_peaks], color='red', label=f'R-peaks {lead}')
        
        # Filter valid R-offsets
        valid_r_offsets = [r_offset for r_offset in r_offsets[lead] if 0 <= r_offset < len(signal)]
        plt.scatter(valid_r_offsets, signal.iloc[valid_r_offsets], color='blue', label=f'R-offsets {lead}')
        
        plt.title(f'ECG Signal with R-peaks and R-offsets for Lead {lead}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../../results/ptb-diagnostic-ecg-database-1.0.0/j-point/{sub_folder}/{fig_title}.png')
    plt.show()


def uplift_signal_to_zero(ecg_signals, r_peak_indices, r_offsets_dict):
    """ Uplifts each signal so that R peaks and offsets are above the zero line. """
    uplifted_signals = {}
    
    for lead, signal in ecg_signals.items():
        # Find the minimum value around R peaks and offsets
        valid_r_peaks = [r_peak for r_peak in r_peak_indices[lead] if 0 <= r_peak < len(signal)]
        valid_r_offsets = [r_offset for r_offset in r_offsets_dict[lead] if 0 <= r_offset < len(signal)]
        
        min_value = np.min(np.concatenate([
            signal.iloc[valid_r_peaks],
            signal.iloc[valid_r_offsets]
        ]))
        
        # Uplift the signal
        uplifted_signals[lead] = signal - min_value + 0.1  # Adding 0.1 to ensure some separation
    
    return uplifted_signals


# +
import matplotlib.pyplot as plt
import numpy as np

def plot_ecg_with_peaks(ecg_signals, r_peak_indices, r_offsets_dict, sub_folder, fig_title):
    plt.figure(figsize=(12, 10))
    
    for i, (lead, signal) in enumerate(ecg_signals.items(), start=1):
        plt.subplot(2, 1, i)  # Create a subplot for each lead
        
        plt.plot(signal, label=f'ECG Lead {lead}')
        
        # Filter valid R-peaks
        valid_r_peaks = [r_peak for r_peak in r_peak_indices[lead] if 0 <= r_peak < len(signal)]
        plt.scatter(valid_r_peaks, signal.iloc[valid_r_peaks], color='red', label=f'R-peaks {lead}')
        
        # Filter valid R-offsets
        valid_r_offsets = [r_offset for r_offset in r_offsets_dict[lead] if 0 <= r_offset < len(signal)]
        plt.scatter(valid_r_offsets, signal.iloc[valid_r_offsets], color='blue', label=f'R-offsets {lead}')
        
        # Calculate average amplitude at R-offsets
        avg_amplitude = np.mean([signal.iloc[r_offset] for r_offset in valid_r_offsets])
        
        # Draw a single horizontal line at the average amplitude of R-offsets
        plt.axhline(y=avg_amplitude, color='g', linestyle='-', linewidth=1, label=f'Avg R-offsets {lead}')
        
        plt.title(f'ECG Signal with R-peaks and R-offsets for Lead {lead}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Add grid lines
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../../results/ptb-diagnostic-ecg-database-1.0.0/j-point/{sub_folder}/{fig_title}.png')
    plt.show()



# -

# # cwt

# + active=""
# for f in range(len(myfiles)):
#     
#     print("\n\n\n-----------------" + str(os.path.basename(myfiles[f])) + "-----------------\n")
#
#     # Read ECG data
#     ECG = read_12_lead_ecg_data(myfiles[f], ECG_time, leads = ['i', 'ii'])
#     # read just first 10000 msecs, 10 secs lead
#     ECG_mv = correct_baseline_wandering(ECG[0:10000], ECG_time[0:10000], plot_result=False)
#     # denoise signal # Apply Butterworth filter to ECG signals
#     filtered_ecgs = butterworth_filter(ECG_mv[0:10000],ECG_time[0:10000], 1000, plot_result=False)
#     # invert signal where required
#     inverted_ECG = read_and_invert_12_lead_ecg_data(filtered_ecgs[0:10000], ECG_time[0:10000], leads = ['i', 'ii'])
#     #print(inverted_ECG)
#     # Find R peaks
#     R_peaks_time = find_R_peaks(inverted_ECG, leads = ['i', 'ii'])
#     
#     r_peak_indices = R_peaks_time
#
#     # Delineate the ECG signals for lead 'i' and 'ii'
#     try:
#         signal_cwt_i, waves_cwt_i = nk.ecg_delineate(inverted_ECG['i'], 
#                                                      R_peaks_time['i'], 
#                                                      sampling_rate=1000, 
#                                                      method="cwt", 
#                                                      show=False, 
#                                                      show_type='bounds_R')
#         print("ECG 'i' delineation successful.")
#     except Exception as e:
#         print(f"Error delineating ECG 'i': {e}")
#     
#     try:
#         signal_cwt_ii, waves_cwt_ii = nk.ecg_delineate(inverted_ECG['ii'], 
#                                                       R_peaks_time['ii'], 
#                                                       sampling_rate=1000, 
#                                                       method="cwt", 
#                                                       show=False, 
#                                                       show_type='bounds_R')
#         print("ECG 'ii' delineation successful.")
#     except Exception as e:
#         print(f"Error delineating ECG 'ii': {e}")
#     # Extract R-offsets
#     r_offsetsi = waves_cwt_i["ECG_R_Offsets"]
#     r_offsetsii = waves_cwt_ii["ECG_R_Offsets"]
#     
#     # Ensure r_offsets is in a dictionary format
#     r_offsets_dict = {'i': r_offsetsi, 'ii':r_offsetsi}
#     # Example usage
#     ecg_signals = {'i': inverted_ECG['i'], 'ii': inverted_ECG['ii']}
#     r_peak_indices = {'i': R_peaks_time['i'], 'ii': R_peaks_time['ii']}
#     r_offsets_dict = {'i': r_offsetsi, 'ii': r_offsetsii}
#     fig_title = str(os.path.basename(myfiles[f]))
#     sub_folder = 'cwt'
#     plot_ecg_with_peaks(ecg_signals, r_peak_indices, r_offsets_dict, sub_folder, fig_title)
# -

# # dwt

# +
for f in range(len(myfiles)):
    
    print("\n\n\n-----------------" + str(os.path.basename(myfiles[f])) + "-----------------\n")

    # Read ECG data
    ECG = read_12_lead_ecg_data(myfiles[f], ECG_time, leads = ['i', 'ii'])
    # read just first 10000 msecs, 10 secs lead
    ECG_mv = correct_baseline_wandering(ECG[0:10000], ECG_time[0:10000], plot_result=False)
    # denoise signal # Apply Butterworth filter to ECG signals
    filtered_ecgs = butterworth_filter(ECG_mv[0:10000],ECG_time[0:10000], 1000, plot_result=False)
    # invert signal where required
    inverted_ECG = read_and_invert_12_lead_ecg_data(filtered_ecgs[0:10000], ECG_time[0:10000], leads = ['i', 'ii'])
    #print(inverted_ECG)

#
    
    # Find R peaks
    R_peaks_time = find_R_peaks(inverted_ECG, leads = ['i', 'ii'])
    
    r_peak_indices = R_peaks_time

    # Delineate the ECG signals for lead 'i' and 'ii'
    try:
        signal_cwt_i, waves_cwt_i = nk.ecg_delineate(inverted_ECG['i'], 
                                                     R_peaks_time['i'], 
                                                     sampling_rate=1000, 
                                                     method="dwt", 
                                                     show=False, 
                                                     show_type='bounds_R')
        print("ECG 'i' delineation successful.")
    except Exception as e:
        print(f"Error delineating ECG 'i': {e}")
    
    try:
        signal_cwt_ii, waves_cwt_ii = nk.ecg_delineate(inverted_ECG['ii'], 
                                                      R_peaks_time['ii'], 
                                                      sampling_rate=1000, 
                                                      method="dwt", 
                                                      show=False, 
                                                      show_type='bounds_R')
        print("ECG 'ii' delineation successful.")
    except Exception as e:
        print(f"Error delineating ECG 'ii': {e}")
    # Extract R-offsets
    r_offsetsi = waves_cwt_i["ECG_R_Offsets"]
    r_offsetsii = waves_cwt_ii["ECG_R_Offsets"]
    
    # Ensure r_offsets is in a dictionary format
    r_offsets_dict = {'i': r_offsetsi, 'ii':r_offsetsi}
    # Example usage
    ecg_signals = {'i': inverted_ECG['i'], 'ii': inverted_ECG['ii']}
    r_peak_indices = {'i': R_peaks_time['i'], 'ii': R_peaks_time['ii']}
    r_offsets_dict = {'i': r_offsetsi, 'ii': r_offsetsii}
    fig_title = str(os.path.basename(myfiles[f]))
    sub_folder = 'dwt'
    plot_ecg_with_peaks(ecg_signals, r_peak_indices, r_offsets_dict ,sub_folder, fig_title)
# -

# # Peak

for f in range(len(myfiles)):
    
    print("\n\n\n-----------------" + str(os.path.basename(myfiles[f])) + "-----------------\n")

    # Read ECG data
    ECG = read_12_lead_ecg_data(myfiles[f], ECG_time, leads = ['i', 'ii'])
    # read just first 10000 msecs, 10 secs lead
    ECG_mv = correct_baseline_wandering(ECG[0:10000], ECG_time[0:10000], plot_result=False)
    # denoise signal # Apply Butterworth filter to ECG signals
    filtered_ecgs = butterworth_filter(ECG_mv[0:10000],ECG_time[0:10000], 1000, plot_result=False)
    # invert signal where required
    inverted_ECG = read_and_invert_12_lead_ecg_data(filtered_ecgs[0:10000], ECG_time[0:10000], leads = ['i', 'ii'])
    #print(inverted_ECG)
    # Find R peaks
    R_peaks_time = find_R_peaks(inverted_ECG, leads = ['i', 'ii'])
    
    r_peak_indices = R_peaks_time

    # Delineate the ECG signals for lead 'i' and 'ii'
    try:
        signal_cwt_i, waves_cwt_i = nk.ecg_delineate(inverted_ECG['i'], 
                                                     R_peaks_time['i'], 
                                                     sampling_rate=1000, 
                                                     method="peak", 
                                                     show=False, 
                                                     show_type='bounds_R')
        print("ECG 'i' delineation successful.")
    except Exception as e:
        print(f"Error delineating ECG 'i': {e}")
    
    try:
        signal_cwt_ii, waves_cwt_ii = nk.ecg_delineate(inverted_ECG['ii'], 
                                                      R_peaks_time['ii'], 
                                                      sampling_rate=1000, 
                                                      method="peak", 
                                                      show=False, 
                                                      show_type='bounds_R')
        print("ECG 'ii' delineation successful.")
    except Exception as e:
        print(f"Error delineating ECG 'ii': {e}")

    print(waves_cwt_i)
    # Extract R-offsets
    r_offsetsi = waves_cwt_i["ECG_R_Offsets"]
    r_offsetsii = waves_cwt_ii["ECG_R_Offsets"]
    
    # Ensure r_offsets is in a dictionary format
    r_offsets_dict = {'i': r_offsetsi, 'ii':r_offsetsi}
    # Example usage
    ecg_signals = {'i': inverted_ECG['i'], 'ii': inverted_ECG['ii']}
    r_peak_indices = {'i': R_peaks_time['i'], 'ii': R_peaks_time['ii']}
    r_offsets_dict = {'i': r_offsetsi, 'ii': r_offsetsii}
    fig_title = str(os.path.basename(myfiles[f]))
    
    plot_ecg_with_peaks(ecg_signals, r_peak_indices, r_offsets_dict, fig_title)

# + active=""
# for f in range(len(myfiles)):
#     print("\n\n\n-----------------" + str(os.path.basename(myfiles[f])) + "-----------------\n")
#     # Read ECG data
#     ECG = read_12_lead_ecg_data(myfiles[f], ECG_time)
#     # read just first 10000 msecs, 10 secs lead
#     ECG_mv = correct_baseline_wandering(ECG[0:10000], ECG_time[0:10000], plot_result=False)
#     # denoise signal # Apply Butterworth filter to ECG signals
#     filtered_ecgs = butterworth_filter(ECG_mv[0:10000],ECG_time[0:10000], 1000, plot_result=False)
#     fig_title = str(os.path.basename(myfiles[f]))
#     plot_12_lead_ecg(filtered_ecgs, ECG_time, fig_title)
#     #plot_12_lead_ecg(ECG[0:10000], ECG_time[0:10000], fig_title)
# -



# + active=""
# for f in range(len(myfiles)):
#     print("\n\n\n-----------------" + str(os.path.basename(myfiles[f])) + "-----------------\n")
#
#     # Read ECG data
#     ECG = read_12_lead_ecg_data(myfiles[f], ECG_time)
#     # read just first 10000 msecs, 10 secs lead
#     ECG_mv = correct_baseline_wandering(ECG[0:10000], ECG_time[0:10000])
#     #print(ECG_mv)
#     inverted_ECG = read_and_invert_12_lead_ecg_data(ECG_mv, ECG_time)
#
#     # Find R peaks
#     R_peaks_time = find_R_peaks(inverted_ECG)
#
#     # Plot all signals in subplots on the same figure
#     fig, axs = plt.subplots(4, 3, figsize=(15, 15))
#     
#     for idx, lead in enumerate(inverted_ECG.columns[:-1]):  # Exclude the last column which is 'time'
#         row = idx // 3
#         col = idx % 3
#         axs[row, col].plot(inverted_ECG[lead], color="navy")
#         axs[row, col].plot(R_peaks_time[lead], inverted_ECG[lead][R_peaks_time[lead]], "o", color="maroon")
#         axs[row, col].set_title(lead)
#         axs[row, col].grid()
#     
#     plt.tight_layout()
#     
#     filename_without_extension = os.path.splitext(os.path.basename(myfiles[f]))[0]
#    # Save the plot with the specified path
#     plt.savefig('../../results/ptb-diagnostic-ecg-database-1.0.0/QRS_visualizations/' + str(filename_without_extension) + '.png')
#
#     plt.show()
