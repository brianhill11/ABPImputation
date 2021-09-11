import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm

sys.path.append("../../")
from abpimputation import project_configs


def ppg_scaling(wave_array: np.ndarray):
    """Scale the PPG signal using min/max scaling, where the most recent 
    systolic/diastolic NIBP measurements are used as max/min values. Format
    for the input array is (n, t, s), where n indexes the window, t indexes 
    time, and s indexes the signal type.

    The min/max scaling is done using the formula:

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min

    For additional information on the scaling, see 
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

    Args:
        wave_array (np.ndarray): Input data matrix, with columns containing
        PPG, Systolic NIBP, Diastolic NIBP. First dimension is window number. 
        second dimension is time, and third dimension is the signal index

    Returns:
        np.ndarray: (n, t, 1) matrix of scaled PPG windows 
    """
    X_std = ((wave_array[:, :, project_configs.ppg_col] - np.expand_dims(wave_array[:, :, project_configs.ppg_col].min(axis=1), -1)) / (np.expand_dims(wave_array[:, :, project_configs.ppg_col].max(axis=1), -1) - np.expand_dims(wave_array[:, :, project_configs.ppg_col].min(axis=1), -1)))

    # we take the median NIBP value in window in case value is updated mid-window
    X_scaled = X_std * (np.median(wave_array[:, :, project_configs.nibp_sys_col], axis=1, keepdims=True) - np.median(wave_array[:, :, project_configs.nibp_dias_col], axis=1, keepdims=True)) + np.median(wave_array[:, :, project_configs.nibp_dias_col], axis=1, keepdims=True)
    return X_scaled

def calc_waveform_feats(window: np.ndarray, debug=False):
    """[summary]

    Args:
        window (np.ndarray): [description]
        debug (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    ecg_sig = window[:, project_configs.ecg_col]
    ppg_sig = window[:, project_configs.ppg_col]
    
    # calculate ECG peaks (Q)
    ecg_peaks, ecg_props = find_peaks(ecg_sig, distance=30, 
        threshold=(0.001, None), width=(1, 10),
        prominence=(1.0, None), wlen=10)
    # calcuate PPG peaks
    ppg_peaks, props = find_peaks(ppg_sig, distance=30, 
        threshold=(0.00001, 0.5), prominence=(0.75, None))

    # estimate heart rate in BPM
    heart_rate = len(ppg_peaks)*60. / (project_configs.window_size / project_configs.sample_freq)
    if debug:
        print("Heart rate:", heart_rate)

    try:
        # make sure first time point is ecg
        ppg_peaks = ppg_peaks[ppg_peaks > ecg_peaks[0]]
        # make sure last point is ppg
        ecg_peaks = ecg_peaks[ecg_peaks < ppg_peaks[-1]]
    except IndexError:
        if debug:
            print(ecg_peaks)
            print(ppg_peaks)
        return np.array([heart_rate, np.nan, np.nan])

    if debug:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(ecg_sig)
        ax[0].scatter(ecg_peaks, ecg_sig[ecg_peaks], c='red')
        ax[0].scatter(ppg_peaks, ecg_sig[ppg_peaks], c='green')

        ax[1].plot(ppg_sig)
        ax[1].scatter(ppg_peaks, ppg_sig[ppg_peaks], c='red')

    try:
        # calculate PAT in seconds
        pat = (ppg_peaks - ecg_peaks) / project_configs.sample_freq
        if debug:
            print(pat)

        # remove any inplausible PAT
        # must be positive PAT
        pat = pat[pat > 0.1]

        # must be less than some threshold
        pat = pat[pat < 1.0]

        # must be at least 2 measurements in window
        if len(pat) < 2:
            pat = np.nan

        if debug:
            print(pat)

    except ValueError as e:
        if debug:
            print(e)
            print(ecg_props)
        return np.array([heart_rate, np.nan, np.nan])

    return np.array([heart_rate, np.median(pat), np.std(pat)])

def create_feature_matrix(X_train: np.ndarray, pickle_dir="abpimputation/models/vnet_32s_mimic"):
    """[summary]

    Args:
        X_train (np.ndarray): [description]
        pickle_dir (str, optional): path to directory containing pickled summary
            statistics used for standardizing features. 
            Defaults to "../models/vnet_32s_mimic".

    Returns:
        np.ndarray: numpy array with additional waveform features added
    """
    # load pickled mean/std and standardize waveform signals
    mean_vals = pickle.load(open(os.path.join(pickle_dir, "mean_vals.pkl"), "rb"))
    std_vals = pickle.load(open(os.path.join(pickle_dir, "std_vals.pkl"), "rb"))
    X_train = (X_train - mean_vals) / std_vals

    # calculate waveform features
    wave_feats = []
    for i in tqdm(range(X_train.shape[0])):
        wave_feats.append(calc_waveform_feats(X_train[i], debug=False))
    wave_feats = np.array(wave_feats)

    # log transform the pulse arrival times
    wave_feats[:, 1] = np.log(wave_feats[:, 1])

    # load pickled waveform feature median/std and scale the waveform features
    wave_feats_median = pickle.load(open(os.path.join(pickle_dir, "wave_feats_median.pkl"), "rb"))
    wave_feats_std = pickle.load(open(os.path.join(pickle_dir, "wave_feats_std.pkl"), "rb"))
    wave_feats_scaled = (wave_feats - wave_feats_median) / wave_feats_std

    # fill in missing waveform feature values
    for i in range(wave_feats_scaled.shape[1]):
        wave_feats_scaled[:, i] = np.nan_to_num(wave_feats_scaled[:, i], nan=wave_feats_median[i])

    # concatenate waveform signals with waveform feature matrix
    X_train = np.concatenate((X_train, np.repeat(np.expand_dims(wave_feats_scaled, axis=1), repeats=project_configs.window_size, axis=1)), axis=2)

    return X_train
