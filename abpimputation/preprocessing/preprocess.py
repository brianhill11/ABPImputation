import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, filtfilt, firwin
from sklearn.preprocessing import StandardScaler, RobustScaler

sys.path.append("../")
from abpimputation import project_configs


def proximity(series: pd.Series):
    """From https://stackoverflow.com/questions/37847150/pandas-getting-the-distance-to-the-row-used-to-fill-the-missing-na-values
    This function is for calculating the number of NaN samples since the most recent valid measurement
    Ex:
              x  prox
        0   NaN     0
        1   NaN     1
        2   NaN     2
        3   3.0     0
        4   NaN     1
        5   NaN     2
        6   NaN     3
        7   5.0     0
        8   NaN     1
        9   NaN     2
        10  NaN     3

    Args:
        series (pd.Series): series with NaN measurements between valid measurements

    Returns:
        pd.Series: series with number of samples since most recent valid measurement
    """
    groupby_idx = series.notnull().cumsum()
    groupby = series.groupby(groupby_idx)
    return groupby.apply(lambda x: pd.Series(range(len(x)))).values


def create_additional_features(input_merged: pd.DataFrame, plot=False):
    """
    Creates additional features using the physiological waveforms (ECG and PPG) 
    and the non-invasive blood pressure (NIBP) measurements. 
    Statistics based on historical NIBP measurements are also added as features.

    Args:
        input_merged (pd.DataFrame): rows are time, columns are 
            (in this order) ECG, PPG, NIBP sys, NIBP dias, NIBP mean
        plot (bool, optional): If True, show debug plots. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe with additional features (columns) added
    """
    simple_nibp_features = set(list(input_merged.columns.values))
    # use prior measurements as additional features
    periods = [5, 10, 15]
    for p in periods:
        for s in project_configs.nibp_column_names:
            col_name_string = "{}_{}_{}".format("median", s, str(p))
            input_merged[col_name_string] = input_merged[s].rolling(p).median()

            col_name_string = "{}_{}_{}".format("std", s, str(p))
            input_merged[col_name_string] = input_merged[s].rolling(p).std()

    # add feature that measures the number of samples since the most recent NIBP value was sampled
    input_merged["prox"] = proximity(input_merged[project_configs.nibp_column_names[-1]])

    derived_nibp_features = list(set(list(input_merged.columns.values)) - simple_nibp_features)
    # since non-invasive BP time may not exactly line up with invasive sampling time, find
    # closest time point for merging
    # impute missing non-invasive measurements by filling forward
    waveform_df = input_merged.fillna(method='ffill')
    # then, if anything is still null, fill with zero
    waveform_df = waveform_df.fillna(0)

    waveform_df.rename(columns=project_configs.signal_column_names, inplace=True)
    print("merged shape:", waveform_df.shape)

    # trim off signal from start/end of record where they are likely hooking up sensors
    # pretrimmed_shape = wav.shape[0]
    # start_index = get_signal_start(wav.iloc[:, 0:4], window_size=400)
    # end_index = get_signal_end(wav.iloc[:, 0:4], window_size=400)
    # print("start index: {} end_index: {}".format(start_index, end_index))
    # wav = wav.iloc[start_index:end_index, :]
    # print("trimmed wave shape: {} (trimmed {}%)".format(wav.shape, (1. - (float(wav.shape[0]) / pretrimmed_shape)) * 100.))
    print(waveform_df.head())

    if plot:
        waveform_df[project_configs.nibp_column_names].plot()
        plt.show()

    waveform_df = waveform_df[["ekg", "ppg", "prox"] + project_configs.nibp_column_names + ["art"]]
    return waveform_df


def filter_wave(data, cutoff, taps, btype, fs=100):
    """Apply filtering to the waveforms 

    Args:
        data ([type]): [description]
        cutoff ([type]): [description]
        taps ([type]): [description]
        btype ([type]): [description]
        fs (int, optional): Sample frequency. Defaults to 100.

    Returns:
        [type]: [description]
    """
    # generates filtered waveform
    if btype == 'lowpass':
        b = firwin(taps, cutoff, window='hamming', fs=fs)
    elif btype == 'bandpass':
        b = firwin(taps, cutoff, window='hamming', pass_zero=False)
    elif btype == 'highpass':
        b = firwin(taps, cutoff, window='hamming', pass_zero=False)
    wav_filter = filtfilt(b, 1, data)
    return wav_filter


def filter_df(wave_df: pd.DataFrame, taps=31, sample_rate=100):
    """Filters the waveform dataframe using low-pass filter to remove 
    high-frequency noise 

    Args:
        wave_df (pd.DataFrame): Input Dataframe with raw waveform signals 
        taps (int, optional): Number of filter taps. Defaults to 31.
        sample_rate (int, optional): Sample frequency. Defaults to 100.

    Returns:
        pd.DataFrame: Dataframe containing filtered waveform signals 
    """
    fmax = sample_rate / 2.
    filtertypes = {'exp1': {'btype': 'lowpass', 'cutoff': 45 / fmax, 
        'median_window': 100, 'median_thresh': 3}}

    wave_df_filtered = pd.DataFrame(index=wave_df.index)

    # cutoff frequency for each waveform
    feature_freq = {"ekg": 16.,
                    "ppg": 16.,
                    "art": 16.}

    #     for feature in wave_df.columns.values:
    for feature in feature_freq.keys():
        wave_df_filtered[feature] = filter_wave(wave_df[feature], 
            feature_freq[feature], taps,
            filtertypes['exp1']['btype'], fs=sample_rate)

    return wave_df_filtered

def preprocess(wave_df: pd.DataFrame):
    """[summary]

    Args:
        wave_df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: Dataframe with additional features added
    """
    wave_df.rename(columns=project_configs.signal_column_names, inplace=True)

    # low-pass filter signal to remove artifacts
    wave_df[["ekg", "ppg", "art"]] = filter_df(wave_df[["ekg", "ppg", "art"]],
        sample_rate=project_configs.sample_freq)

    # create additional features from signal
    wave_df = create_additional_features(wave_df)
    return wave_df