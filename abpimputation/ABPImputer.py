import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

import abpimputation.project_configs as project_configs
from abpimputation.preprocessing.features import ppg_scaling, create_feature_matrix

def batch_custom_loss(y_true, y_pred):
    # multiply ABP with mask to get sys/dias BP values
    true_bp_values = y_true[:, :, 0] * y_true[:, :, 1]
    # # # multiply pred ABP with mask to get pred sys/dias BP values
    pred_bp_values = y_pred[:, :, 0] * y_true[:, :, 1]
    # # # calculate mean squared distance between true/pred values
    bp_error = MeanSquaredError(true_bp_values, pred_bp_values)
    # get mean absolute error between all true/pred values
    mse = MeanSquaredError(y_true[:, :, 0], y_pred[:, :, 0])
    return mse + bp_error

class ABPImputer(BaseEstimator):

    def __init__(self, demo_param='demo', pickle_dir="abpimputation/models/vnet_32s_mimic"):
        
        model_dir = "abpimputation/models/vnet_32s_mimic"
        
        assert os.path.exists(pickle_dir)
        assert os.path.exists(model_dir)

        print("Loading saved model...")
        self.model = tf.keras.models.load_model(model_dir, 
            custom_objects={
                "batch_custom_loss": batch_custom_loss,
            })
        print(self.model.summary())
        self.demo_param = demo_param

        # load pickled mean/std and standardize waveform signals
        self.abp_mean_val = pickle.load(open(os.path.join(pickle_dir, "abp_mean_val.pkl"), "rb"))
        self.abp_std_val  = pickle.load(open(os.path.join(pickle_dir, "abp_std_val.pkl"), "rb"))
        
        # self.waveform_filter = WaveformFilter()
        self.samples_per_window = int(32 * project_configs.sample_freq)

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # prepare input matrix

        # create mask vectors for sys/dias points

        # train model

        return

    def predict(self, X: np.ndarray):
        # Check is fit had been called
        # check_is_fitted(self)
        print("X_mean:", X.mean(axis=(0, 1), keepdims=True))

        # prepare input matrix by creating features and scaled PPG signal
        self._prepare_matrix(X)        

        # use trained model to predict residual waveform
        y_pred_raw = self.model.predict(self.X_train_feat)

        # add residual waveform to scaled PPG 
        res = self._inverse_scale_abp(y_pred_raw)[:, :, 0]
        y_pred = self.scaled_ppg + res
        
        return y_pred 

    def reshape_to_window(self, X: np.ndarray):
        # make sure there is additional dimension for transformation
        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=-1)
        # trim record to multiple of window size 
        num_windows = X.shape[0] // self.samples_per_window
        X = X[:num_windows*self.samples_per_window]
        print(f"New shape: {X.shape}")

        # reshape record into non-overlapping windows 
        print(f"Reshaping record into {num_windows} windows")
        if len(X.shape) > 1:
            newshape = (num_windows, self.samples_per_window, X.shape[-1])
        elif len(X.shape) == 1:
            newshape = (num_windows, self.samples_per_window)
        print(f"Expected shape: {newshape}")
        X_train_reshaped = np.reshape(X, newshape=newshape)
        print(f"New {len(newshape)}D shape: {X_train_reshaped.shape}")
        return X_train_reshaped

    def _prepare_matrix(self, X: np.ndarray):
        # TODO:? if fit has not been called, calculate outlier values

        # reshape record into non-overlapping windows 
        X_train_reshaped = self.reshape_to_window(X)

        # TODO:? filter outlier values, save valid index mask as member variable

        # calculate scaled PPG, save as member variable
        self.scaled_ppg = ppg_scaling(X_train_reshaped)
        print(f"Scaled PPG shape: {self.scaled_ppg.shape}")

        # calculate waveform features and standardize data
        self.X_train_feat = create_feature_matrix(X_train_reshaped)

        # make sure input waveforms are padded along time axis for convolutions
        pad_size = 4
        self.X_train_feat = np.pad(self.X_train_feat, 
            ((0, 0), (pad_size, pad_size), (0, 0)), 
            mode='edge')

        return self.X_train_feat, self.scaled_ppg

    def _inverse_scale_abp(self, X: np.ndarray):
        """Inverse scale some standardized values by multiplying by the
        standard deviation and adding back the mean value

        Args:
            X (np.ndarray): Standardized input data array

        Returns:
            np.ndarray: Rescaled output data array
        """
        return (X * self.abp_std_val) + self.abp_mean_val
