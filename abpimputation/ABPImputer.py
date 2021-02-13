import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from WaveformFilter import WaveformFilter

class ABPImputer(BaseEstimator):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
        self.waveform_filter = WaveformFilter()

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



