# %% imports 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import abpimputation.project_configs as project_configs
from abpimputation.ABPImputer import ABPImputer
from abpimputation.preprocessing.preprocess import preprocess
from abpimputation.preprocessing.features import create_feature_matrix
# %% get list of files

input_dir = "/Users/bhizzle/Downloads/cvc_test"
save_dir = "/Users/bhizzle/Downloads/abp_imputation_test"
base_filename = "p015046-2199-01-08-22-03_merged.csv.gz"
f = os.path.join(input_dir, base_filename)
os.makedirs(save_dir, exist_ok=True)

data = pd.read_csv(f, index_col=0)
data.head()

# %% preprocess the data 
preprocessed_data = preprocess(data)
preprocessed_data.head()

# %% split into input/target data
y_true = preprocessed_data["art"]
X_train = preprocessed_data.iloc[:, :-1]
# add in pseudo-time axis 
# y_true = np.expand_dims(y_true, axis=-1)
X_train = np.expand_dims(X_train, axis=1)

# %%
print(X_train.shape)
print(y_true.shape)

# %% instantiate ABPImputer
abp = ABPImputer()

# %% generate predicted ABP waveform
y_pred = abp.predict(X_train)
y_pred.shape

# %%
# _, ax = plt.subplots(1, 1, figsize=(12, 6))
# plt.plot(y_pred[10], label="y_pred")
# plt.plot(abp.scaled_ppg[10], label="scaled PPG", alpha=0.7)
# plt.legend()
# plt.plot()

# %%
# y_true_reshaped = abp.reshape_to_window(y_true)
# plt.plot(y_true_reshaped[10])

# %% flatten imputed ABP signal and add column for data
y_pred_flattened = y_pred.flatten()
data["imputed_abp"] = np.nan
data["imputed_abp"].iloc[:y_pred_flattened.shape[0]]  = y_pred_flattened

# # %%
# fig, ax = plt.subplots(2, 1, figsize=(12, 12))
# data["art"].plot(ax=ax[0])
# data["imputed_abp"].plot(ax=ax[1])
# plt.show()

# %% write new file to disk
data.to_csv(os.path.join(save_dir, base_filename), header=True, index=True)
