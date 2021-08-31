import os

home_dir = os.environ["HOME"]
#project_dir = os.path.join(home_dir, "Downloads/test_mimic_project")
project_dir = os.path.join("/Volumes/External/mimic_v8_32s")
#project_dir = os.path.join("/Volumes/Sammy/mimic_v9_4s")

#scratch_dir = "/u/scratch/b/blhill"
#project_dir = os.path.join(scratch_dir, "mimic_v2")
#project_dir = os.path.join(os.environ["SCRATCH"], "mimic_v2")

# if not os.path.exists(project_dir):
#     os.makedirs(project_dir)

# should either be "mimic" or "uci" - need to handle different file naming conventions and file formats
dataset = "mimic"

# number of samples per second for waveforms
sample_freq = 100.

# raw waveform file directory (should be generated after running download_mimic_raw.sh)
raw_data_dir = os.path.join(os.environ["PWD"], "../../data/raw/physionet.org/physiobank/database/")

# directory containing .csv.gz files, one per patient
preprocessed_data_dir = os.path.join(project_dir, "mimic_preprocessed")

# directory containing NIBP .csv.gz files, one per patient
nibp_data_dir = os.path.join(project_dir, "nibp_data")

#################################################
# Training config
#################################################

# directory containing training windows
train_dir = os.path.join(project_dir, "train_patients")
#train_dir = os.path.join(project_dir, "train_windows_v2")
#train_dir = "/Volumes/External/mimic_v4/train_windows_v2"
# directory containing validation windows
val_dir = os.path.join(project_dir, "val_patients")
#val_dir = os.path.join(project_dir, "val_windows_v2")
# directory containing testing windows
#test_dir = os.path.join(project_dir, "test_patients")
test_dir = os.path.join(project_dir, "test_windows_v2")
#test_dir = "/Volumes/External/mimic_v4/test_windows_v2"

X_scaler_pickle = "../../models/train_X_scaler.pkl"
y_scaler_pickle = "../../models/train_y_scaler.pkl"

add_nibp_noise = False

#################################################
# Window filtering configs
#################################################

# directory to save window files
window_save_dir = os.path.join(project_dir, "train_mimic_windows")

# window filtering stats file directory
stats_file_dir = os.path.join(project_dir, "window_stats")

# number of samples in window
window_size = 3200

# number of samples in window stride
window_stride = int(window_size/2)

# max windows per numpy array file
max_windows_per_file = 50

# number of samples in larger window to use for finding/fixing lag between pleth wave and art wave
shift_window_size = 3200

# column name mapping for waveforms
signal_column_names = {"II": "ekg", "PLETH": "ppg", "ABP": "art"}
# column names for non-invasive blood pressure measurements
# should be in the order: systolic, diastolic, mean
nibp_column_names = ["pseudo_NIBP_sys_5min", "pseudo_NIBP_dias_5min", "pseudo_NIBP_mean_5min"]

# order of signals in .npy files
ecg_col = 0
ppg_col = 1
prox_col = 2
nibp_sys_col = 3
nibp_dias_col = 4
nibp_mean_col = 5
abp_col = -1  # should be 6

# directory to save images of invalid windows
invalid_window_images = os.path.join(project_dir, "invalid_window_images")
# directory to save images of valid windows
valid_window_images = os.path.join(project_dir, "valid_window_images")

#################################################
# Model Configs
#################################################

# padding if input window to avoid issues with convolution at endpoints
padding_size = 4

# number of windows in batch
batch_size = 32

# number of filters in CNN layers
num_filters = 128


#################################################
# Plotting Configs
#################################################
axis_label_font_size = 16
title_font_size = 18
axis_tick_label_size = 16
