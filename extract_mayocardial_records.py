# + active=""
# """
# This script processes ECG data from a specified directory, identifying and categorizing patient records
# based on a specific diagnosis, such as "Myocardial infarction." It performs the following tasks:
#
# 1. **Import Required Libraries**: Imports necessary libraries for handling files, warnings, and data manipulation.
#
# 2. **Read Header Files**: Reads '.hea' files from patient directories to extract information about diagnoses.
#
# 3. **Diagnosis Identification**: Searches for specific diagnoses within the content of the '.hea' files.
#
# 4. **Folder Management**:
#    - Moves patient folders associated with "Myocardial infarction" to a new directory named accordingly.
#    - Deletes the original folders after moving.
#    - Organizes the remaining patient folders into a "Control group" directory.
#
# 5. **Directory Cleanup**: Ensures all relevant patient records are organized based on the diagnosis, with clear folder naming.
# """
# -


import wfdb
from wfdb import processing
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import pandas as pd
# import path_management file from src dir
import warnings
warnings.filterwarnings("ignore")


# +
# find the format of header file to know how the diagnosis class is provided 

# +
def read_heas_from_directory(directory):
    heas = []
    for filename in os.listdir(directory):
        if filename.endswith(".hea"):
            hea_path = os.path.join(directory, filename)
            with open(hea_path, 'r') as file:
                hea_content = file.read()
                heas.append(hea_content)
    return heas


directory_path = "../../data/ptb-diagnostic-ecg-database-1.0.0/patient001"

heas = read_heas_from_directory(directory_path)
for hea in heas:
    print(hea)


# -

# # Move records with mayacardial infarction into new folder with clear name Mayocardial Infarction and delete empty folders where match found

# +
def move_patient_folder(source_directory, destination_directory, patient_folder, diagnosis):
    """
    Moves a patient's folder from a source directory to a destination directory,
    renaming it to include a diagnosis identifier. The function also removes the
    original folder after moving.

    Parameters:
    - source_directory (str): The directory where the original patient folder is located.
    - destination_directory (str): The directory where the patient folder should be moved to.
    - patient_folder (str): The name of the patient folder to be moved.
    - diagnosis (str): A string representing the diagnosis, which will be appended to the patient folder name in the destination directory.

    Functionality:
    - Constructs the full paths for the source patient folder and the destination folder with the diagnosis appended.
    - Ensures that the destination folder exists or creates it.
    - Moves all files and subdirectories from the source folder to the destination folder.
    - Deletes the original patient folder from the source directory after all items have been moved.
    - Prints confirmation messages for folder movement and deletion.
    """
    source_patient_path = os.path.join(source_directory, patient_folder)
    destination_diagnosis_folder = os.path.join(destination_directory, patient_folder + "_" + diagnosis)
    os.makedirs(destination_diagnosis_folder, exist_ok=True)
    
    # Move all files and subdirectories from source to destination
    for item in os.listdir(source_patient_path):
        source_item_path = os.path.join(source_patient_path, item)
        destination_item_path = os.path.join(destination_diagnosis_folder, item)
        shutil.move(source_item_path, destination_item_path)
    
    print("Folder moved:", destination_diagnosis_folder)
    # Delete the original patient folder after moving
    shutil.rmtree(source_patient_path)
    print("Original folder deleted:", source_patient_path)

def read_heas_from_directory(directory):
    """
    Reads all '.hea' files from a specified directory (including its subdirectories)
    and returns their contents along with their file paths.

    Parameters:
    - directory (str): The root directory to search for '.hea' files.

    Functionality:
    - The function traverses the specified directory and its subdirectories using `os.walk`.
    - For each '.hea' file found:
        - Constructs the full path to the file.
        - Opens and reads the file's contents line by line.
        - Appends a tuple containing the root directory, filename, and file contents to a list.
    - Returns a list of tuples, where each tuple contains:
        - The directory path where the '.hea' file was found.
        - The filename of the '.hea' file.
        - The content of the '.hea' file as a list of lines.
    """
    heas = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".hea"):
                hea_path = os.path.join(root, filename)
                with open(hea_path, 'r') as file:
                    hea_content = file.readlines()
                    heas.append((root, filename, hea_content))
    return heas

def find_diagnosis_in_heas(hea_files, diagnosis):
    """
    Searches for a specific diagnosis within the content of '.hea' files and returns the result.

    Parameters:
    - hea_files (list of tuples): A list of tuples, each containing:
        - root (str): The directory path where the '.hea' file was found.
        - filename (str): The name of the '.hea' file.
        - content (list of str): The content of the '.hea' file as a list of lines.
    - diagnosis (str): The diagnosis to search for within the '.hea' files.
    """
    for root, filename, content in hea_files:
        for line in content:
            if line.startswith("# Reason for admission:") and diagnosis.lower() in line.lower():
                print(f"Diagnosis found in file: {filename}")
                return True, os.path.basename(root)
    return False, None


source_directory = "../../data/ptb-diagnostic-ecg-database-1.0.0"
destination_directory = "../../data/ptb-diagnostic-ecg-database-1.0.0"

diagnosis = "Myocardial infarction"

patient_folders = [folder for folder in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, folder))]

for patient_folder in patient_folders:
    heas = read_heas_from_directory(os.path.join(source_directory, patient_folder))
    for root, filename, content in heas:
        found, patient_folder = find_diagnosis_in_heas([(root, filename, content)], diagnosis)
        if found:
            move_patient_folder(source_directory, destination_directory, patient_folder, diagnosis)
            break
    else:
        print(f"Diagnosis not found in any files in folder: {patient_folder}")

# -

# # Move maycardial folders into one folder and rest into folder called control group

# +
source_directory = "../../data/ptb-diagnostic-ecg-database-1.0.0"

# Define destination directories
destination_directory_infarction = os.path.join(os.path.dirname(source_directory), "Myocardial infarction")
destination_directory_control = os.path.join(source_directory, "Control group")

# Create the destination directories if they don't exist
os.makedirs(destination_directory_infarction, exist_ok=True)
os.makedirs(destination_directory_control, exist_ok=True)

# Iterate through each item in the source directory
for item in os.listdir(source_directory):
    source_path = os.path.join(source_directory, item)
    # Check if the item is a directory
    if os.path.isdir(source_path):
        # Check if the directory name contains "Myocardial infarction"
        if "_Myocardial infarction" in item:
            # Move to Myocardial infarction directory outside Control group
            destination_path = os.path.join(destination_directory_infarction, item)
        else:
            # Move to Control group inside the source directory
            destination_path = os.path.join(destination_directory_control, item)
        
        # Move the directory to the appropriate destination
        try:
            shutil.move(source_path, destination_path)
            print(f"Moved {item} to {destination_path}")
        except Exception as e:
            print(f"Error moving {item}: {e}")

print("All folders moved.")

# -

# # collect files from multiple sub folders into one folder

def extract_and_cleanup_files(source_directory):
    """
    Extracts all files from subdirectories within the specified source directory
    and moves them into a single folder named 'all_files'. After moving the files,
    the original subdirectories are removed.

    Parameters:
    - source_directory (str): The path to the source directory containing the subdirectories and files.
    """
    # Define the destination directory
    destination_directory = os.path.join(source_directory, "all_files")

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    # Iterate through each subdirectory and file in the source directory
    for root, dirs, files in os.walk(source_directory, topdown=False):
        for file in files:
            # Construct the full path to the file
            file_path = os.path.join(root, file)
            
            # Skip moving the file if it's already in the destination directory
            if os.path.basename(root) == "all_files":
                continue
            
            # Move the file to the destination directory
            try:
                shutil.move(file_path, os.path.join(destination_directory, file))
                print(f"Moved: {file}")
            except Exception as e:
                print(f"Error moving {file}: {e}")
        
        # After moving files, remove the directory if it's not the destination directory
        if root != destination_directory:
            try:
                os.rmdir(root)
                print(f"Removed empty directory: {root}")
            except OSError as e:
                print(f"Error removing directory {root}: {e}")

    print("All files have been moved to the 'all_files' folder, and subfolders have been removed.")


# Move Mayocardial
source_directory = "../../data/Myocardial infarction/"
extract_and_cleanup_files(source_directory)

# Move control group
source_directory = "../../data/ptb-diagnostic-ecg-database-1.0.0/Control group/"
extract_and_cleanup_files(source_directory)


# # Catergorize records based on age and gender

def categorize_records(source_directory):
    """
    Categorizes and moves .hea, .dat, and .xyz files based on age and sex.
    
    - Males aged 40 or younger -> 'male_40_or_younger'
    - Males older than 40 -> 'male_greater_than_40'
    - All females -> 'females'
    - Records with missing or 'n/a' age -> 'age_missing'
    - Records with missing or 'n/a' sex -> 'sex_missing'

    Parameters:
    - source_directory (str): Path to the directory containing .hea, .dat, and .xyz files.
    """
    # Define the destination directories
    male_40_or_younger_dir = os.path.join(source_directory, "male_40_or_younger")
    male_greater_than_40_dir = os.path.join(source_directory, "male_greater_than_40")
    females_dir = os.path.join(source_directory, "females")
    age_missing_dir = os.path.join(source_directory, "age_missing")
    sex_missing_dir = os.path.join(source_directory, "sex_missing")
    
    # Create destination directories if they don't exist
    os.makedirs(male_40_or_younger_dir, exist_ok=True)
    os.makedirs(male_greater_than_40_dir, exist_ok=True)
    os.makedirs(females_dir, exist_ok=True)
    os.makedirs(age_missing_dir, exist_ok=True)
    os.makedirs(sex_missing_dir, exist_ok=True)
    
    # Iterate through each .hea file in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith(".hea"):
            hea_path = os.path.join(source_directory, filename)
            dat_path = os.path.splitext(hea_path)[0] + ".dat"  # Associated .dat file
            xyz_path = os.path.splitext(hea_path)[0] + ".xyz"  # Associated .xyz file

            # Read .hea file to find age and sex
            with open(hea_path, 'r') as file:
                content = file.readlines()
                age = None
                sex = None
                for line in content:
                    if line.startswith("# age:"):
                        age_str = line.split(":")[1].strip()
                        if age_str.isdigit():  # Check if age is a valid number
                            age = int(age_str)
                        else:
                            age = "missing"  # Use a special marker for missing or invalid age
                            break
                    elif line.startswith("# sex:"):
                        sex_str = line.split(":")[1].strip().lower()
                        sex = sex_str if sex_str in ["male", "female"] else "missing"

            # Determine the destination folder based on age and sex
            if age == "missing":
                destination_dir = age_missing_dir
            elif sex == "missing" or sex is None:
                destination_dir = sex_missing_dir
            elif sex == "male":
                if age <= 40:
                    destination_dir = male_40_or_younger_dir
                else:
                    destination_dir = male_greater_than_40_dir
            elif sex == "female":
                destination_dir = females_dir
            else:
                print(f"Skipping {filename} due to unexpected sex value: {sex}")
                continue
            
            # Move .hea, .dat, and .xyz files to the appropriate directory
            try:
                shutil.move(hea_path, os.path.join(destination_dir, filename))
                if os.path.exists(dat_path):  # Check if associated .dat file exists
                    shutil.move(dat_path, os.path.join(destination_dir, os.path.basename(dat_path)))
                if os.path.exists(xyz_path):  # Check if associated .xyz file exists
                    shutil.move(xyz_path, os.path.join(destination_dir, os.path.basename(xyz_path)))
                print(f"Moved {filename}, {os.path.basename(dat_path)}, and {os.path.basename(xyz_path)} to {destination_dir}")
            except Exception as e:
                print(f"Error moving {filename}, {os.path.basename(dat_path)}, or {os.path.basename(xyz_path)}: {e}")


# categorize Control group
source_directory = "../../data/ptb-diagnostic-ecg-database-1.0.0/Control group/all_files"
categorize_records(source_directory)

# categorize myocardial group
source_directory = "../../data/Myocardial infarction/all_files"
categorize_records(source_directory)
