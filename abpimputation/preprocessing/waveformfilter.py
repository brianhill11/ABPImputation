import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import tensorflow as tf

import abpimputation.project_configs as project_configs


class WaveformFilter(BaseEstimator):
    """
    Class for filtering a (n, t, c) dimension matrix, where
    n is the number of windows
    t is the number of time samples
    c is the number of channels

    Note that the channels are expected to be a in particular order, as
    set in the project_configs.py file.

    """
    def __init__(self,
                 max_min_from_cuff=5,
                 max_num_flat_abp=2,
                 lower_quantile=0.001,
                 upper_quantile=0.999,
                 drift_threshold=40,
                 pulse_pressure_threshold=70,
                 ppg_qi_model_file="../../PPG_QI/weights/weights.ppgqi.hdf5",
                 ppg_scaler_pkl_file="../../PPG_QI/weights/ppg_qi_scaler.pkl"):
        """

        :param max_min_from_cuff: max number of minutes from last NIBP measurement
        :param max_num_flat_abp: max number of consecutive ABP samples with no change (flat signal)
        :param lower_quantile: lower quantile to use for filtering outlier windows based on mean signal value
        :param upper_quantile: upper quantile to use for filtering outlier windows based on mean signal value
        :param drift_threshold: max difference between max ABP value in window and most recent systolic NIBP value
        :param pulse_pressure_threshold: max difference between min and max ABP value in window
        """

        self.ppg_qi_model_file = ppg_qi_model_file
        self.ppg_scaler_pkl_file = ppg_scaler_pkl_file
        # threshold determined on MIMIC validation data such that precision=0.95
        self.ppg_qi_threshold = 0.8108692169189453

        self.max_min_from_cuff = max_min_from_cuff
        self.max_num_flat_abp = max_num_flat_abp
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.drift_threshold = drift_threshold
        self.pulse_pressure_threshold = pulse_pressure_threshold

        # dictionary mapping filtering function to boolean mask, where mask is True if window is valid
        self.mask_dict = {}

    def fit(self, X):
        """

        :param np.array X:
        :return: None
        """
        self._calc_outlier_values(X)
        return

    def transform(self, X, verbose=False):
        """

        :param X:
        :param bool verbose: if True, show filtering statistics
        :return: np.array X after filtering
        """
        # Check is fit had been called
        check_is_fitted(self, attributes=["outlier_values_"])

        # remove outlier values
        self._filter_outlier_values(X)

        # remove windows too far from most recent cuff measurement
        self._filter_far_from_cuff(X)

        # remove windows that are missing NIBP
        self._filter_missing_NIBP(X)

        # remove windows with flat ABP waveform
        self._filter_flat_ABP(X)

        # remove windows where NIBP-ABP drift too large
        self._filter_drift(X)

        # remove windows where pulse pressure is outside of reasonable range
        self._filter_pulse_pressure(X)

        # filter based on PPG QI value
        self._filter_ppg_qi(X)

        shape_pre_filter = X.shape[0]
        X = self._remove_invalid_samples(X)
        if verbose:
            print(self.get_filtering_stats())
            print("Removed {} windows with filtering".format(shape_pre_filter - X.shape[0]))
        return X

    def fit_transform(self, X):
        """

        :param X:
        :return:
        """
        self.fit(X)
        return self.transform(X)

    def get_filtering_stats(self):
        """
        Return dictionary containing valid masks for each filtering step
        :return: dict mask_dict
        """
        return self.mask_dict

    def print_filtering_stats(self):
        """
        Use filtering masks to generate stats about valid windows
        :return: None
        """
        print(pd.DataFrame.from_dict(self.mask_dict).mean())
        return None

    def _remove_invalid_samples(self, X):
        """
        Use filtering masks to remove invalid samples
        :param X:
        :return: np.array X without invalid windows
        """
        # create a filter mask containing all True values
        valid_windows = np.array([True]*X.shape[0])
        # for each filter stage, element-wise AND with the filter mask to get final filter mask
        for filter_name, mask in self.mask_dict.items():
            assert (mask.dtype == bool), "filter mask must be of type bool"
            valid_windows = np.logical_and(valid_windows, mask)
        # use final filter mask to remove invalid samples
        return X[valid_windows]

    def _calc_outlier_values(self, X):
        """

        :param X:
        :return: None
        """
        channels = [project_configs.ecg_col, project_configs.ppg_col, project_configs.abp_col]
        # dictionary mapping channel [ecg, ppg, abp] to an upper/lower bound tuple for the mean signal value
        self.outlier_values_ = {}
        # for each channel, get upper and lower bound for mean signal value
        for c in channels:
            upper_bound = np.nanquantile(X[:, :, c].mean(axis=1), q=self.upper_quantile)
            lower_bound = np.nanquantile(X[:, :, c].mean(axis=1), q=self.lower_quantile)
            self.outlier_values_[c] = (lower_bound, upper_bound)

    def _filter_outlier_values(self, X):
        """

        :param X:
        :return: None
        """
        # for each channel, use lower and upper bound for mean of signals to find outlier windows
        for channel, bounds in self.outlier_values_.items():
            lower_bound, upper_bound = bounds[0], bounds[1]
            valid_indices_lb = X[:, :, channel].mean(axis=1) >= lower_bound
            valid_indices_ub = X[:, :, channel].mean(axis=1) <= upper_bound
            self.mask_dict["filter_outlier_values_lb_{}".format(channel)] = valid_indices_lb
            self.mask_dict["filter_outlier_values_ub_{}".format(channel)] = valid_indices_ub

    def _filter_ppg_qi(self, X, qi_aggregation_func=min, ppg_window_size=400):
        """
        TODO: clean this function up
        :param X:
        :param qi_aggregation_func: function for aggregating PPG QI values across a window
        :param ppg_window_size: input window size for PPG QI model
        :return: None
        """
        if X.shape[0] == 0:
            return X

        # load PPG QI model from file
        ppg_qi = tf.keras.models.load_model(self.ppg_qi_model_file)

        # load PPG QI scaler object
        ppg_qi_scaler = pickle.load(open(self.ppg_scaler_pkl_file, "rb"))

        # calculate number of PPG QI windows within input window
        # for example, if we have 32s windows, we have 8 PPG QI predictions
        # per window/row
        num_windows = int(project_configs.window_size / ppg_window_size)

        ppg_window_preds = []
        for i in range(num_windows):
            # print(int(i * ppg_window_size), int(i * ppg_window_size + ppg_window_size))
            ppg_window_preds.append(ppg_qi.predict(np.expand_dims(ppg_qi_scaler.transform(
                X[:, int(i * ppg_window_size):int(i * ppg_window_size + ppg_window_size), project_configs.ppg_col]),
                                                                  -1))[:, 0])

        # stack each pred, and take min (default) across windows (rows) to get worst case PPG QI in each window
        ppg_qi_preds = np.apply_along_axis(qi_aggregation_func, axis=1, arr=np.column_stack(ppg_window_preds))

        valid_window_idx = np.array(ppg_qi_preds) > self.ppg_qi_threshold

        self.mask_dict["filter_ppg_qi"] = valid_window_idx

    def _filter_far_from_cuff(self, X):
        """

        :param X:
        :return: None
        """
        # get time from cuff (proximity) in seconds
        time_from_cuff = X[:, :, project_configs.prox_col].mean(axis=1) / project_configs.sample_freq

        # find all rows where mean prox is less than threshold
        max_num_secs_from_cuff = self.max_min_from_cuff * 60
        valid_samples_from_cuff = time_from_cuff <= max_num_secs_from_cuff

        self.mask_dict["filter_far_from_cuff"] = valid_samples_from_cuff

    def _filter_missing_NIBP(self, X):
        """

        :param X:
        :return: None
        """
        # Remove windows with no NIBP measurements
        valid_sys = np.median(X[:, :, project_configs.nibp_sys_col], axis=1) > 0
        valid_dias = np.median(X[:, :, project_configs.nibp_dias_col], axis=1) > 0
        self.mask_dict["filter_missing_NIBP"] = np.logical_and(valid_sys, valid_dias)

    def _filter_flat_ABP(self, X):
        """

        :param X:
        :return: None
        """
        # Remove windows where the ABP signal is flat for more than 1 sample
        valid_indices = []
        for i in tqdm(range(X.shape[0])):
            # get count of number of consecutive samples where difference between samples is zero
            ediffs = np.count_nonzero(np.ediff1d(X[i, :, project_configs.abp_col]) == 0.0)
            # 1 can happen by chance but any more than that seems to be artifacts
            if ediffs < self.max_num_flat_abp:
                valid_indices.append(True)
            else:
                valid_indices.append(False)

        self.mask_dict["filter_flat_ABP"] = np.array(valid_indices)

    def _filter_drift(self, X):
        """

        :param X:
        :return: None
        """
        # remove windows with unreasonable drift
        valid_indices = np.abs(np.max(X[:, :, project_configs.abp_col], axis=1) - np.median(
            X[:, :, project_configs.nibp_sys_col], axis=1)) < self.drift_threshold
        self.mask_dict["filter_drift"] = valid_indices

    def _filter_pulse_pressure(self, X):
        """

        :param X:
        :return: None
        """
        # remove windows with unreasonable "pulse pressure" values

        valid_indices = np.max(X[:, :, project_configs.abp_col], axis=1) - np.min(X[:, :, project_configs.abp_col],
                                                                                  axis=1) < self.pulse_pressure_threshold
        self.mask_dict["filter_pulse_pressure"] = valid_indices
