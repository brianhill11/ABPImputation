import sys
import os
import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout
from keras.layers import Lambda
from keras.layers import Input, Reshape
from keras.layers import LSTM
from keras import backend as K

sys.path.append("../../")
from abpimputation import project_configs


def create_model_sideris(trainable=True, save_dir=None):
    if save_dir is None:
        save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dropout_rate = 0.5
    # initial bias value
    c = 0.01
    batch_norm_epsilon = 0.2
    depth_multiplier = 1

    num_channels = 28
    num_input_channels = 6 + 3
    num_static_vars = num_channels - 6
    ekg_index = 0
    pleth_index = 1
    sys_abp_index = -3
    dias_abp_index = -2
    mean_abp_index = -1
    # default parameters for RMSprop
    optimizer = RMSprop(lr=0.001, rho=0.9, clipnorm=0.5)

    inputs = Input(shape=(project_configs.window_size + 2 * project_configs.padding_size, num_input_channels))

    wave_tensors = Lambda(lambda x: x[:, project_configs.padding_size:-project_configs.padding_size, project_configs.ppg_col],
                          output_shape=(project_configs.window_size,))(inputs)
    wave_tensors = Reshape((-1, 1))(wave_tensors)

    num_units = 128
    unroll = False

    output = LSTM(num_units, return_sequences=True, unroll=unroll)(wave_tensors)
    output = Dropout(rate=dropout_rate)(output)
    wave = Dense(1, activation=None, name="wave")(output)


    model = Model(inputs=inputs, outputs=[wave])
    model.compile(loss={"wave": "mse"},
                  metrics=['mse'],
                  optimizer=optimizer)

    model.summary()

    return model