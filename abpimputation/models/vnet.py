import sys
import os
import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.optimizers import RMSprop, Nadam
from keras.initializers import Constant
from keras.losses import mean_squared_error
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers import Lambda, Concatenate
from keras.layers import Conv1D,  Conv2DTranspose
from keras.layers import Input, Reshape, Cropping1D, SpatialDropout1D, ReLU
from keras.layers import Add
from keras.layers import LSTM
from keras import regularizers
from keras import backend as K

sys.path.append("../../")
from abpimputation import project_configs


def batch_custom_loss(y_true, y_pred):
    """Custom loss function that calculates mean squared error (MSE) between
    the true and predicted waveform, and the mean squared error at the systolic
    and diastolic points in the waveform, as determined by the binary mask 
    encoding the positions of the systolic/diastolic points.  


    Args:
        y_true (np.array): A 3-dimensional numpy array, where the first index
        is batch, the second index is time, and the third index is channel. 
        The first channel is assumed to be the raw waveform. The second channel
        is assumed to be the binary mask that encodes the systolic/diastolic 
        peak locations.
        y_pred (np.array): A 3-dimensional numpy array, where the first index
        is batch, the second index is time, and the third index is channel.
        The predicted waveform should be located in the first (and only) channel

    Returns:
        float: mean squared error from waveform and BP points
    """
    # multiply ABP with mask to get sys/dias BP values
    true_bp_values = y_true[:, :, 0] * y_true[:, :, 1]
    # # # multiply pred ABP with mask to get pred sys/dias BP values
    pred_bp_values = y_pred[:, :, 0] * y_true[:, :, 1]
    # # # calculate mean absolute distance between true/pred values
    bp_error = mean_squared_error(true_bp_values, pred_bp_values)
    # get mean absolute error between all true/pred values
    mse = mean_squared_error(y_true[:, :, 0], y_pred[:, :, 0])
    return mse + bp_error


def create_model_vnet(trainable=True, save_dir=None):
    if save_dir is None:
        save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dropout_rate = 0.2
    # initial bias value
    c = 0.01
    batch_norm_epsilon = 0.2
    depth_multiplier = 1
    num_channels = 32
    if project_configs.window_size == 200:
        num_convolutions = (1, 2, 3)
    else:
        num_convolutions = (1, 2, 3, 3)
    num_levels = len(num_convolutions)

    bottom_convolutions = 3
    kernel_size = 5

    num_input_channels = 6 + 5
    num_static_vars = num_channels - 6
    ekg_index = 0
    pleth_index = 1
    sys_abp_index = -3
    dias_abp_index = -2
    mean_abp_index = -1
    
    optimizer = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    inputs = Input(shape=(project_configs.window_size + 2 * project_configs.padding_size,
                          9))
    
    wave_tensors = Lambda(lambda x: x[:, :, 0:])(inputs)

    def convolution_block(layer_input, n_convolutions):
        x = layer_input
        try:
            n_channels = int(x.shape[-1][-1])
        except TypeError:
            n_channels = int(x.shape[-1])
        for i in range(n_convolutions):
            x = Conv1D(n_channels, kernel_size=kernel_size, activation=None, padding='same',
                       bias_initializer=Constant(c))(x)
            if i == n_convolutions - 1:
                x = Add()([x, layer_input])
            x = BatchNormalization(epsilon=batch_norm_epsilon)(x)
            x = ReLU()(x)
            x = SpatialDropout1D(rate=dropout_rate)(x)
        return x

    def convolution_block_2(layer_input, fine_grained_features, n_convolutions):
        x = Concatenate(axis=-1)([layer_input, fine_grained_features])
        try:
            n_channels = int(layer_input.shape[-1][-1])
        except TypeError:
            n_channels = int(layer_input.shape[-1])
        if n_convolutions == 1:
            x = Conv1D(n_channels, kernel_size=kernel_size, activation=None, padding='same',
                       bias_initializer=Constant(c))(x)
            x = BatchNormalization(epsilon=batch_norm_epsilon)(x)
            layer_input = BatchNormalization(epsilon=batch_norm_epsilon)(layer_input)
            x = Add()([x, layer_input])
            x = BatchNormalization(epsilon=batch_norm_epsilon)(x)
            x = ReLU()(x)
            x = SpatialDropout1D(rate=dropout_rate)(x)
            return x

        x = Conv1D(n_channels, kernel_size=kernel_size, activation=None, padding='same',
                   bias_initializer=Constant(c))(x)
        x = BatchNormalization(epsilon=batch_norm_epsilon)(x)
        x = ReLU()(x)
        x = SpatialDropout1D(rate=dropout_rate)(x)

        for i in range(1, n_convolutions):
            x = Conv1D(n_channels, kernel_size=kernel_size, activation=None, padding='same',
                       bias_initializer=Constant(c))(x)
            if i == n_convolutions - 1:
                layer_input = BatchNormalization(epsilon=batch_norm_epsilon)(layer_input)
                x = Add()([x, layer_input])
            x = BatchNormalization(epsilon=batch_norm_epsilon)(x)
            x = ReLU()(x)
            x = SpatialDropout1D(rate=dropout_rate)(x)

        return x

    def down_convolution(x, factor, kernel_size):
        try:
            n_channels = int(x.shape[-1][-1])
        except TypeError:
            n_channels = int(x.shape[-1])
        x = Conv1D(n_channels*factor, kernel_size=kernel_size, strides=factor, padding='same',
                   bias_initializer=Constant(c))(x)
        return x

    def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
        """
            FROM: https://stackoverflow.com/questions/44061208/how-to-implement-the-conv1dtranspose-in-keras
            input_tensor: tensor, with the shape (batch_size, time_steps, dims)
            filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
            kernel_size: int, size of the convolution kernel
            strides: int, convolution step size
            padding: 'same' | 'valid'
        """
        try:
            k = Lambda(lambda k: K.expand_dims(k, axis=2), output_shape=(input_tensor.shape[-1][-2], 1,
                                                                     input_tensor.shape[-1][-1]))(input_tensor)
        except TypeError:
            k = Lambda(lambda k: K.expand_dims(k, axis=2), output_shape=(input_tensor.shape[-2], 1,
                                                                     input_tensor.shape[-1]))(input_tensor)

        k = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(k)
        k = Lambda(lambda k: K.squeeze(k, axis=2))(k)
        return k

    def up_convolution(x, factor, kernel_size):
        try:
            n_channels = int(x.shape[-1][-1])
        except TypeError:
            n_channels = int(x.shape[-1])

        x = Conv1DTranspose(x, n_channels // factor, kernel_size=kernel_size, padding='same',
                            strides=factor)
        return x

    output = Conv1D(int(16), kernel_size=5, bias_initializer=Constant(c), padding='same',
                             activation=None,
                             input_shape=(project_configs.batch_size,
                                          project_configs.window_size + 2 * project_configs.padding_size,
                                          num_channels),
                             trainable=trainable)(wave_tensors)
    output = Cropping1D(cropping=(4, 4))(output)
    output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
    output = ReLU()(output)

    features = list()
    # compression
    for l in range(num_levels):
        output = convolution_block(output, n_convolutions=num_convolutions[l])
        features.append(output)
        output = down_convolution(output, factor=2, kernel_size=kernel_size)
        output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
        output = ReLU()(output)

    # bottom
    output = convolution_block(output, bottom_convolutions)

    # decompression
    for l in reversed(range(num_levels)):
        f = features[l]
        output = up_convolution(output, factor=2, kernel_size=kernel_size)
        output = BatchNormalization(epsilon=batch_norm_epsilon)(output)
        output = ReLU()(output)

        output = convolution_block_2(output, f, num_convolutions[l])

    wave = Conv1D(1, kernel_size=1, use_bias=True, bias_initializer=Constant(c), padding='same', activation=None,
                  trainable=True, activity_regularizer=regularizers.l2(0.0005), name="wave")(output)


    model = Model(inputs=inputs, outputs=[wave])
    model.compile(loss=batch_custom_loss, metrics=['mse'], optimizer=optimizer)

    model.summary()

    return model

