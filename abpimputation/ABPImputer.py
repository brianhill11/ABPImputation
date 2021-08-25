import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

# from WaveformFilter import WaveformFilter

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

    def __init__(self, demo_param='demo'):
        print("Loading saved model...")
        model_dir = "abpimputation/models/vnet_32s_mimic"
        self.model = tf.keras.models.load_model(model_dir, 
            custom_objects={
                "batch_custom_loss": batch_custom_loss,
            })
        print(self.model.summary())
        self.demo_param = demo_param
        # self.waveform_filter = WaveformFilter()

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # prepare input matrix

        # create mask vectors for sys/dias points

        # train model

        return

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # prepare input matrix

        # use trained model to

    def _prepare_matrix(self, X):
        # if fit has not been called, calculate outlier values


        # filter outlier values, save valid index mask as member variable

        # calculate scaled PPG, save as member variable

        # calculate waveform features

        # standardize the data
        pass



