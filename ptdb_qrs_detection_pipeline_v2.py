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
#from scipy.signal import find_peaks
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
#from scipy import signal, datasets
import pandas as pd
import shutil
import math
# -

from scipy import signal

import warnings
warnings.filterwarnings('ignore')

# move to src directory from the current directory to import QTC_nomogram
current_directory = os.getcwd()
src_directory = os.path.join(current_directory, '..', '..', 'src')
sys.path.append(src_directory)
# --

# + active=""
# def move_files_to_one_folder(source_folder, destination_folder):
#     # Ensure the destination folder exists
#     os.makedirs(destination_folder, exist_ok=True)
#     
#     # Walk through all subdirectories in the source folder
#     for root, dirs, files in os.walk(source_folder):
#         for file in files:
#             source_file = os.path.join(root, file)
#             destination_file = os.path.join(destination_folder, file)
#             
#             # If the file already exists in the destination, rename the file to avoid overwriting
#             if os.path.exists(destination_file):
#                 base, extension = os.path.splitext(file)
#                 counter = 1
#                 new_destination_file = f"{base}_{counter}{extension}"
#                 while os.path.exists(os.path.join(destination_folder, new_destination_file)):
#                     counter += 1
#                     new_destination_file = f"{base}_{counter}{extension}"
#                 destination_file = os.path.join(destination_folder, new_destination_file)
#             
#             # Move the file
#             shutil.move(source_file, destination_file)
#             print(f"Moved: {source_file} to {destination_file}")
#
#     print("All files have been moved successfully.")
#
# # Example usage:
# source_folder = '/Users/user/ECG-X_DataReader/data/ptb-diagnostic-ecg-database-1.0.0/Control group'
# destination_folder = '/Users/user/ECG-X_DataReader/data/all_files_control_group'
# move_files_to_one_folder(source_folder, destination_folder)

# + active=""
# import wfdb
# import pandas as pd
# import os
#
# def convert_wfdb_to_csv(record_name, output_csv):
#     # Read the WFDB record
#     record = wfdb.rdrecord(record_name)
#     
#     # Convert to a pandas DataFrame
#     df = pd.DataFrame(record.p_signal, columns=record.sig_name)
#     
#     # Save to CSV
#     df.to_csv(output_csv, index=False)
#     
#     # Reading header and other metadata
#     with open(f'{record_name}.hea', 'r') as f:
#         metadata = f.read()
#     
#     # Save metadata as a separate file or append to CSV
#     metadata_file = f'{output_csv}_metadata.txt'
#     with open(metadata_file, 'w') as f:
#         f.write(metadata)
#     
#     print(f"Converted {record_name} to {output_csv}")
#     print(f"Metadata saved to {metadata_file}")
#
# def process_all_records_in_directory(input_directory, output_directory):
#     # Ensure the output directory exists
#     os.makedirs(output_directory, exist_ok=True)
#     
#     for filename in os.listdir(input_directory):
#         if filename.endswith(".hea"):
#             # Remove the .hea extension to get the base name
#             record_name = os.path.splitext(filename)[0]
#             record_path = os.path.join(input_directory, record_name)
#             
#             output_csv = os.path.join(output_directory, f'{record_name}.csv')
#             
#             # Check if corresponding .dat file exists
#             if os.path.isfile(f'{record_path}.dat'):
#                 try:
#                     convert_wfdb_to_csv(record_path, output_csv)
#                 except Exception as e:
#                     print(f"Failed to convert {record_name}: {e}")
#             else:
#                 print(f"Skipping {record_name} as corresponding .dat file is missing")
#
# # Example usage:
# input_directory = '/Users/user/ECG-X_ST/data/all_files_control_group'
# output_directory = '/Users/user/ECG-X_ST/data/control_group_csv_format'
#
# process_all_records_in_directory(input_directory, output_directory)
#
# -

def generate_ecg_time_column(sampling_rate=1000, num_samples=38401):
    """
    Generate a time column as a DataFrame for ECG data.

    Parameters:
    sampling_rate (int): Sampling rate in samples per second (default is 1000).
    num_samples (int): Number of samples to generate (default is 38401).

    Returns:
    pd.DataFrame: DataFrame containing the time column with 'ms' as the column name.
    """
    # Calculate the time interval
    time_interval = 1 / sampling_rate

    # Generate the time column
    time_column = [i * time_interval for i in range(num_samples)]

    # Convert the time column to a DataFrame
    ECG_time = pd.DataFrame({'ms': time_column})

    return ECG_time



# +

mydir = "/Users/user/ECG-X_ST/data/control_group_csv_format"
myfiles = glob.glob(mydir + "/*.csv")


# -

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


# # Step 3: Noise Removal

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


# # Step 4: Invert signal

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

# # Step 5: Find QRS

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


# # Step 6: Detect R Peak Offset (J-point) Using dwt

def detect_r_peak_offset(inverted_ECG, R_peaks_time):
    """
    Delineate the ECG signals for lead 'i' and 'ii' and extract R-offsets.

    This function uses discrete wavelet transform (DWT) to delineate the ECG signals 
    for the given leads and extract the R-offsets from the delineated waves.

    Parameters
    ----------
    inverted_ECG : pandas.DataFrame
        A DataFrame containing the inverted ECG data for the leads.
    R_peaks_time : list
        A list of times (in seconds) where R-peaks are detected in the ECG signal.

    Returns
    -------
    r_offsetsi : list
        A list containing the times of the R-offsets for the given ECG lead.
    """
    
    try:
        signal_dwt, waves_dwt = nk.ecg_delineate(inverted_ECG, 
                                                 R_peaks_time, 
                                                 sampling_rate=1000, 
                                                 method="dwt", 
                                                 show=False, 
                                                 show_type='bounds_R')
        print("ECG 'ii' delineation successful.")
    except Exception as e:
        print(f"Error delineating ECG: {e}")
    # Extract R-offsets
    r_offsetsi = waves_dwt["ECG_R_Offsets"]

    return r_offsetsi


# # Step 7: Detect and Uplift Custom Isoelectric Line

def detect_and_uplift_isoelectric_line(signal, r_offsets):
    """
    Calculates the average amplitude at the R-offsets and uplifts the signal by this average amplitude.
    
    Parameters:
    signal (pd.Series): The ECG signal.
        A pandas Series containing the ECG signal data.
    r_offsets (list): List of R-offset indices.
        A list of indices representing the R-offsets in the ECG signal.

    Returns:
    pd.Series: The uplifted ECG signal.
        A pandas Series containing the ECG signal uplifted by the average amplitude calculated from R-offsets.
    """
    valid_r_offsets = [r_offset for r_offset in r_offsets if 0 <= r_offset < len(signal)]
    avg_amplitude = np.mean([signal.iloc[r_offset] for r_offset in valid_r_offsets])
    
    uplifted_signal = signal - avg_amplitude
    return uplifted_signal


# # Step 8: Plot ECG with Peaks

# +
def plot_ecg_with_peaks(uplifted_ecg_signals, r_peak_indices, r_offsets_dict, sub_folder, fig_title):
    """
    Plots uplifted ECG signals with R-peaks and R-offsets.
    
    Parameters:
    uplifted_ecg_signals (dict): Dictionary containing uplifted ECG signals.
        A dictionary where keys are lead identifiers and values are pandas Series of uplifted ECG signals.
    r_peak_indices (dict): Dictionary containing R-peak indices.
        A dictionary where keys are lead identifiers and values are lists of R-peak indices.
    r_offsets_dict (dict): Dictionary containing R-offsets.
        A dictionary where keys are lead identifiers and values are lists of R-offset indices.
    sub_folder (str): The sub-folder where the plot will be saved.
        A string specifying the sub-folder path relative to the current directory.
    fig_title (str): The title of the figure.
        A string specifying the title of the plot.
    """
    plt.figure(figsize=(12, 10))
    
    for i, (lead, uplifted_signal) in enumerate(uplifted_ecg_signals.items(), start=1):
        plt.subplot(2, 1, i)  # Create a subplot for each lead
        
        plt.plot(uplifted_signal, label=f'ECG Lead {lead}')
        
        # Filter valid R-peaks
        valid_r_peaks = [r_peak for r_peak in r_peak_indices[lead] if 0 <= r_peak < len(uplifted_signal)]
        plt.scatter(valid_r_peaks, uplifted_signal.iloc[valid_r_peaks], color='red', label=f'R-peaks {lead}')
        
        # Filter valid R-offsets
        valid_r_offsets = [r_offset for r_offset in r_offsets_dict[lead] if 0 <= r_offset < len(uplifted_signal)]
        plt.scatter(valid_r_offsets, uplifted_signal.iloc[valid_r_offsets], color='blue', label=f'R-offsets {lead}')
        
        plt.title(f'ECG Signal with R-peaks and R-offsets for Lead {lead}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Add grid lines
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../../results/{sub_folder}/{fig_title}.png')
    plt.show()


def process_and_plot_ecg(ecg_signals, r_peak_indices, r_offsets_dict, sub_folder, fig_title):
    """
    Processes ECG signals by uplifting them and plots the uplifted signals with R-peaks and R-offsets.
    
    Parameters:
    ecg_signals (dict): Dictionary containing ECG signals.
        A dictionary where keys are lead identifiers and values are pandas Series containing ECG signal data.
    r_peak_indices (dict): Dictionary containing R-peak indices.
        A dictionary where keys are lead identifiers and values are lists of R-peak indices.
    r_offsets_dict (dict): Dictionary containing R-offsets.
        A dictionary where keys are lead identifiers and values are lists of R-offset indices.
    sub_folder (str): The sub-folder where the plot will be saved.
        A string specifying the sub-folder path relative to the current directory.
    fig_title (str): The title of the figure.
        A string specifying the title of the plot.
    """
    uplifted_ecg_signals = {}
    for lead, signal in ecg_signals.items():
        uplifted_signal = detect_and_uplift_isoelectric_line(signal, r_offsets_dict[lead])
        uplifted_ecg_signals[lead] = uplifted_signal
    
    plot_ecg_with_peaks(uplifted_ecg_signals, r_peak_indices, r_offsets_dict, sub_folder, fig_title)


# -

# Simply keeping this func to check how signal evolve after each opertaion i-e wandering correction , noise removal etc
def plot_ecg(signal_data, title):
    """
    Plot the first two ECG signal leads (i and ii).

    Args:
    - signal_data (dict): Dictionary containing ECG signal data. Keys are lead names (str) and values are signal arrays (array-like).
    - title (str): Title of the plot.

    Returns:
    - None
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot lead i
    axs[0].plot(signal_data['i'], label='Lead i')
    axs[0].set_ylabel('ECG Amplitude')
    axs[0].set_title('Lead i')

    # Plot lead ii
    axs[1].plot(signal_data['ii'], label='Lead ii')
    axs[1].set_ylabel('ECG Amplitude')
    axs[1].set_title('Lead ii')

    axs[1].set_xlabel('Sample')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()



# # Step 9: Acceptance Testing

# +
sampling_rate = 1000
num_samples = 38401
ECG_time = generate_ecg_time_column(sampling_rate, num_samples)

# Assuming `myfiles` is a list of file paths
for f in range(len(myfiles)):
    
    print("\n\n\n-----------------" + str(os.path.basename(myfiles[f])) + "-----------------\n")

    # Read ECG data
    ECG = read_12_lead_ecg_data(myfiles[f], ECG_time, leads=['i', 'ii'])
    plot_ecg(ECG, "ECG") #1
    # Correct baseline wandering for the first 10 seconds (10000 ms)
    ECG_mv = correct_baseline_wandering(ECG[0:10000], ECG_time[0:10000], plot_result=False)
    plot_ecg(ECG_mv, "Baseline Wandering Correction") #2
    # Apply Butterworth filter to ECG signals noise removal
    filtered_ecgs = butterworth_filter(ECG_mv[0:10000], ECG_time[0:10000], 1000, plot_result=False)
    plot_ecg(filtered_ecgs, "Noise Removal") #3
    # Invert signal where required
    inverted_ECG = read_and_invert_12_lead_ecg_data(filtered_ecgs[0:10000], ECG_time[0:10000], leads=['i', 'ii'])
    plot_ecg(inverted_ECG, "Inverted ECG") #4
    # Find R peaks
    R_peaks_time = find_R_peaks(inverted_ECG, leads=['i', 'ii'])
    
    # Detect R-offsets
    r_offsetsi = detect_r_peak_offset(inverted_ECG['i'], R_peaks_time['i'])
    r_offsetsii = detect_r_peak_offset(inverted_ECG['ii'], R_peaks_time['ii'])
    
    # Ensure r_offsets_dict is in dictionary format
    r_offsets_dict = {'i': r_offsetsi, 'ii': r_offsetsii}
    
    # Prepare data for plotting
    ecg_signals = {'i': inverted_ECG['i'], 'ii': inverted_ECG['ii']}
    r_peak_indices = {'i': R_peaks_time['i'], 'ii': R_peaks_time['ii']}
    r_offsets_dict = {'i': r_offsetsi, 'ii': r_offsetsii}
    
    # Define sub-folder and figure title
    fig_title = str(os.path.basename(myfiles[f]))
    sub_folder = 'control_group'
    
    # Plot ECG with peaks and offsets
    process_and_plot_ecg(ecg_signals, r_peak_indices, r_offsets_dict, sub_folder, fig_title)
