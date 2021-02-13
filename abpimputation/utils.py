import os
import numpy as np
import pandas as pd
import glob
import shutil
import pickle
import subprocess as sp
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from tqdm import tqdm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import train_test_split

import abpimputation.project_configs as project_configs


def bland_altman_plot_v2(y_true, y_pred,
                         axis_lim=[50, 200],
                         figsize=(8, 8),
                         sd_limit=1.96,
                         scatter_kwds=None,
                         mean_line_kwds=None,
                         limit_lines_kwds=None,
                         title_string="Bland-Altman: {} +/- {}"):
    means = np.mean([y_true, y_pred], axis=0)
    diffs = np.array(y_true) - np.array(y_pred)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'gray'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    p = sns.scatterplot(x=means, y=diffs, ax=ax)

    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

    # Annotate mean line with mean difference.
    ax.annotate('Mean:\n{:.2f}'.format(np.round(mean_diff, 2)),
                xy=(0.99, 0.5),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=14,
                xycoords='axes fraction')

#     half_ylim = (1.5 * sd_limit) * std_diff
#     ax.set_ylim(mean_diff - half_ylim,
#                 mean_diff + half_ylim)

    limit_of_agreement = sd_limit * std_diff
    lower = mean_diff - limit_of_agreement
    upper = mean_diff + limit_of_agreement
    for j, lim in enumerate([lower, upper]):
        ax.axhline(lim, **limit_lines_kwds)
    ax.annotate('-SD{}: {:.2f}'.format(sd_limit, np.round(lower, 2)),
                xy=(0.99, 0.07),
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=14,
                xycoords='axes fraction')
    ax.annotate('+SD{}: {:.2f}'.format(sd_limit, np.round(upper, 2)),
                xy=(0.99, 0.92),
                horizontalalignment='right',
                fontsize=14,
                xycoords='axes fraction')

    # set axis limits
    plot_lim = [-60, 60]
    p.set_ylim(plot_lim)
    p.set_xlim(axis_lim)
    # set axis ticks
    p.set_yticks(np.arange(plot_lim[0], plot_lim[1] + 1, 15))
    p.set_xticks(np.arange(axis_lim[0], axis_lim[1] + 1, 25))
    ax.tick_params(labelsize=13)
    # set axis labels
    axis_label_font_size = 14
    p.set_ylabel("Invasive - Predicted Arterial Pressure [mmHg]", fontsize=axis_label_font_size)
    p.set_xlabel("(Invasive + Predicted Arterial Pressure)/2 [mmHg]", fontsize=axis_label_font_size)
    # add number of points to plot
    ax.legend(["N={}".format(len(y_true))], loc='upper left')
    # add title
    title_font_size = 16
    ax.set_title(title_string.format(np.round(mean_diff, 2), np.round(std_diff, 2)), fontsize=title_font_size)
    plt.tight_layout()


def bland_altman_plot(m1, m2,
                      sd_limit=1.96,
                      ax=None,
                      scatter_kwds=None,
                      mean_line_kwds=None,
                      limit_lines_kwds=None):
    """
    Bland-Altman Plot.
    A Bland-Altman plot is a graphical method to analyze the differences
    between two methods of measurement. The mean of the measures is plotted
    against their difference.
    Parameters
    ----------
    m1, m2: pandas Series or array-like
    sd_limit : float, default 1.96
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted will be
                       md - sd_limit * sd, md + sd_limit * sd
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences.
        If sd_limit = 0, no limits will be plotted, and the ylimit of the plot
        defaults to 3 standard deviatons on either side of the mean.
    ax: matplotlib.axis, optional
        matplotlib axis object to plot on.
    scatter_kwargs: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method
    mean_line_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
    limit_lines_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method
   Returns
    -------
    ax: matplotlib Axis object
    """

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

    means = np.mean([m1, m2], axis=0)
    diffs = np.array(m1) - np.array(m2)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    if ax is None:
        ax = plt.gca()

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'gray'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    ax.scatter(means, diffs, **scatter_kwds)
    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

    # Annotate mean line with mean difference.
    ax.annotate('Mean:\n{}'.format(np.round(mean_diff, 1)),
                xy=(0.99, 0.5),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=14,
                xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff

        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)
        ax.annotate('-SD{}: {}'.format(sd_limit, np.round(lower, 1)),
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=14,
                    xycoords='axes fraction')
        ax.annotate('+SD{}: {}'.format(sd_limit, np.round(upper, 1)),
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    fontsize=14,
                    xycoords='axes fraction')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff

    # bland-altman y-axis limits
    plot_lim = [-60, 60]
    ax.set_ylim(plot_lim)
    
    ax.set_ylabel('Difference between 2 measures', fontsize=15)
    ax.set_xlabel('Average of 2 measures', fontsize=15)
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    return ax


def get_art_peaks(sig, max_peaks_per_sec=4, min_peaks_per_sec=1, plot=False, distance=36, sample_freq=100):
    """
    This function gets the max/min peaks that should correspond to systolic and diastolic BP

    X: 1D array
    max_peaks_per_second: max number of peaks we should see in a 1 sec
    window. Default is 4 since 220/60 = 3.6667 which is approx 4
    min_peaks_per_second: min number of peaks we should see in a 1 sec
    window. Default is 1, but this may not be a valid assumption
    distance: minimum number of samples required between peaks
    """
    indices_max, props_max = find_peaks(sig, distance=distance, threshold=(None, 5.0), prominence=(20, None))
    indices_min, props_min = find_peaks(-sig, distance=distance, threshold=(None, 5.0), prominence=(10, None))
    return list(indices_max), list(indices_min)


def align_lists(a, b):
    """
    align two lists by finding minimum difference between values in sliding window
    :param list a: first list of indices
    :param list b: second list of indices
    """
    l = [a, b]
    if len(a) != len(b):
        biggest_list = np.argmax([len(x) for x in l])
        smallest_list = np.argmin([len(x) for x in l])
    else:
        return a, b
    smallest_list_len = len(l[smallest_list])
    num_slide = len(l[biggest_list]) - len(l[smallest_list]) + 1
    diffs = []
    for i in range(num_slide):
        diff = np.sum(np.array(l[biggest_list][i:i+smallest_list_len]) - np.array(l[smallest_list]))
        diffs.append(diff)
    diffs = np.abs(diffs)
    index = np.argmin(diffs)
    if l[smallest_list] == b:
        return l[biggest_list][index:index+smallest_list_len], l[smallest_list]
    elif l[smallest_list] == a:
        return l[smallest_list], l[biggest_list][index:index+smallest_list_len]
    else:
        raise ValueError


def train_val_test_split_mimic(data_dir="/Volumes/External/",
                               split_file_dir="data/",
                               create_file_list=False,
                               remove_old_dirs=False,
                               train_split=0.8,
                               val_split=0.1):
    """
    Function for creating train/validation/test splits by patient, and moving files into
    corresponding directories

    :param data_dir: directory containing all of the project directories
    :param split_file_dir: location to save/load patient split files to/from
    :param create_file_list: set to True if you want to create a new train/val/test split
    :param remove_old_dirs: set to True if you want to remove the old directories containing the split data
    :param train_split: fraction of entire dataset to use for training
    :param val_split: fraction of entire dataset to use for validation
    :return: None
    """

    # we want the intersection of patient lists for all window sizes
    dirs = glob.glob(os.path.join(data_dir, "mimic_v*_*"))

    # this is for creating the train/val/test patient splits
    if create_file_list:
        patient_sets = {}
        for d in dirs:
            patient_sets[d] = []
            # get directories with window data
            window_dirs = glob.glob(os.path.join(d, "*_mimic_windows"))
            for w in window_dirs:
                print(w)
                files = np.unique([os.path.basename(x).split("-")[0] for x in glob.glob(os.path.join(w, "*.npy"))])
                print(files[0:10])
                patient_sets[d] = patient_sets[d] + list(files)

        for k, v in patient_sets.items():
            print(k, len(set(v)))

        # take intersection of patient lists so that we have same patients for all window sizes
        intersection_patients = set.intersection(*[set(x) for x in patient_sets.values()])
        print("Found {} total patients".format(len(intersection_patients)))

        # first split entire dataset into train and test
        train_ids, test_ids = train_test_split(list(intersection_patients), train_size=train_split)
        # then split train set into train and validation
        train_ids, val_ids = train_test_split(train_ids, train_size=1.-(val_split*(1./train_split)))

        print("Num train: {} ({:03f}%)".format(len(train_ids), len(train_ids) / len(intersection_patients) * 100))
        print("Num val:   {}  ({:03f}%)".format(len(val_ids), len(val_ids) / len(intersection_patients) * 100))
        print("Num test:  {} ({:03f}%)".format(len(test_ids), len(test_ids) / len(intersection_patients) * 100))

        # write list of patients for each split to file
        with open(os.path.join(split_file_dir, "train_patients.txt"), "w") as f:
            for p in train_ids:
                f.write("{}\n".format(p))

        with open(os.path.join(split_file_dir, "val_patients.txt"), "w") as f:
            for p in val_ids:
                f.write("{}\n".format(p))

        with open(os.path.join(split_file_dir, "test_patients.txt"), "w") as f:
            for p in test_ids:
                f.write("{}\n".format(p))
        print("Split files created...")

    print("Moving window files to train/val/test directories")
    # use patient split files to move files into train/val/test directories
    patient_ids = {}
    for i in ["train_patients.txt", "val_patients.txt", "test_patients.txt"]:
        with open(os.path.join(split_file_dir, i), "r") as f:
            patient_ids[os.path.splitext(i)[0]] = [l.strip() for l in f.readlines()]

    for k, v in patient_ids.items():
        print(k, len(v))

    for d in tqdm(dirs):
        for k, v in tqdm(patient_ids.items()):
            source_dir = os.path.join(d, "all_mimic_windows")
            dest_dir = os.path.join(d, k)

            if remove_old_dirs:
                print("Removing {}".format(dest_dir))
                shutil.rmtree(dest_dir)

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            #         print("source_dir:", source_dir)
            #         print("dest_dir:", dest_dir)

            for p in tqdm(v):
                file_list = glob.glob(os.path.join(source_dir, p + "*"))
                for f in file_list:
                    #                 shutil.copy(f, dest_dir)
                    if not os.path.exists(os.path.join(dest_dir, os.path.basename(f))):
                        shutil.move(f, dest_dir)


def train_test_split_mimic(train_dir, test_dir, train_split=0.8):
    """
    This code is for creating train/test sets from MIMIC data, such that
    patients are either only in train or test set, not both

    :param train_dir: directory that contains training windows
    :param test_dir:  directory to move testing windows to
    :param train_split: fraction of windows to keep for training
    :return: None
    """
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # get list of patients
    train_files = glob.glob(os.path.join(train_dir, "*"))
    patients = set([os.path.basename(i).split("_")[0] for i in train_files])
    print(sorted(patients))

    # get patient IDs and dates from file name
    patient_ids = [i.split("-", 1)[0] for i in patients]
    patient_dates = [i.split("-", 1)[1] for i in patients]
    patient_df = pd.DataFrame.from_dict({"ptid": patient_ids, "date": patient_dates})

    # split into train/test patient lists
    train_patients = patient_df.ptid.drop_duplicates().sample(frac=train_split)
    test_patients = patient_df[~patient_df.ptid.isin(train_patients)].ptid.unique()

    print("train patients shape:", train_patients.shape)
    print("test patients shape:", test_patients.shape)

    train_patient_df = patient_df[patient_df.ptid.isin(train_patients)]
    print(train_patient_df.shape)
    test_patient_df = patient_df[patient_df.ptid.isin(test_patients)]
    print(test_patient_df.shape)

    # move files for the test patients to the test directory
    for index, row in tqdm(test_patient_df.iterrows()):
        file_string = "-".join([row["ptid"], row["date"]])
        file_list = glob.glob(os.path.join(train_dir, file_string + "*"))
        for f in file_list:
            shutil.move(f, test_dir)


def train_test_split_uci(train_dir, test_dir, train_split=0.8):
    """
    This code is for creating train/test sets from UCI data, such that
    patients are either only in train or test set, not both

    :param train_dir: directory that contains training windows
    :param test_dir:  directory to move testing windows to
    :param train_split: fraction of windows to keep for training
    :return: None
    """
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # get list of patients
    train_files = glob.glob(os.path.join(train_dir, "*"))
    patients = set([os.path.basename(i).split("_")[0] for i in train_files])
    print(sorted(patients))

    # get patient IDs and dates from file name
    patient_df = pd.DataFrame.from_dict({"ptid": list(patients)})

    # split into train/test patient lists
    train_patients = patient_df.ptid.drop_duplicates().sample(frac=train_split)
    test_patients = patient_df[~patient_df.ptid.isin(train_patients)].ptid.unique()

    print("train patients shape:", train_patients.shape)
    print("test patients shape:", test_patients.shape)

    train_patient_df = patient_df[patient_df.ptid.isin(train_patients)]
    print(train_patient_df.shape)
    test_patient_df = patient_df[patient_df.ptid.isin(test_patients)]
    print(test_patient_df.shape)

    # move files for the test patients to the test directory
    for index, row in tqdm(test_patient_df.iterrows()):
        file_string = row["ptid"]
        file_list = glob.glob(os.path.join(train_dir, file_string + "*"))
        for f in file_list:
            shutil.move(f, test_dir)


def remove_empty_files(path):
    """
    Use linux 'find' command to find and delete empty files (0 bytes)
    :param path: directory to search for empty files
    :return: None
    """
    if os.path.exists(path):
        print("Removing empty files from {}".format(path))
        output = sp.check_output(['find', path, '-size', '0', '-exec', 'rm', '{}', ';'])
        print(output)
    else:
        print("{} does not exist! Creating directory...".format(path))
        os.makedirs(path)
    return None


def log_project_configs(save_dir):
    """
    Store information necessary to recreate results
    :param save_dir: path to directory for saving files
    :return: None
    """
    return None
    shutil.copy("../project_configs.py", save_dir)
    shutil.copy("abp_model.py", save_dir)


def load_scaler_objects(X_scaler_pickle, y_scaler_pickle):
    """
    Function for loading scaler objects
    :param X_scaler_pickle: path to X scaler pickle object
    :param y_scaler_pickle: path to y scaler pickle object
    :return: sklearn's StandardScaler object for input, target data
    """
    # read in scaler objects
    X_train_scaler = pickle.load(open(X_scaler_pickle, "rb"))
    y_train_scaler = pickle.load(open(y_scaler_pickle, "rb"))

    return X_train_scaler, y_train_scaler


def get_demo_df(train_demo_file="../visualization/train_demographics.csv",
                val_demo_file="../visualization/val_demographics.csv",
                test_demo_file="../visualization/test_demographics.csv",
                dump_scaler_imputer=False):
    """
    Function for loading demographic data. Code assumes that the following columns with the
     following names are present: subject_id,age,height,weight,bmi,sex

    Age is expected to be in years, height in cm, weight in kg, and sex coded as 0 for male or 1 for female
    :param str train_demo_file: path to training data demographic data file
    :param str val_demo_file: path to validation data demographic data file
    :param str test_demo_file: path to test data demographic data file
    :param bool dump_scaler_imputer: set to True to generate pickled scaler & imputer objects for demo data
    :return: pandas.DataFrame with scaled and imputed demographic data
    """
    # load demographic data for train/test sets
    train_df = pd.read_csv(train_demo_file, sep=",", header=0, index_col=0)
    val_df = pd.read_csv(val_demo_file, sep=",", header=0, index_col=0)
    test_df = pd.read_csv(test_demo_file, sep=",", header=0, index_col=0)

    # make sure columns are in the correct order
    col_order = ["age", "height", "weight", "bmi", "sex"]
    train_df = train_df[col_order]
    val_df = val_df[col_order]
    test_df = test_df[col_order]

    # convert sex string to integer code
    try:
        train_df["sex"] = train_df["sex"].apply(lambda x: 0 if x.lower() == "m" else 1)
        val_df["sex"] = val_df["sex"].apply(lambda x: 0 if x.lower() == "m" else 1)
        test_df["sex"] = test_df["sex"].apply(lambda x: 0 if x.lower() == "m" else 1)
    # unless value is already a float
    except AttributeError as e:
        pass

    # if dumping current objecfts
    if dump_scaler_imputer:
        print("Dumping new demographic data scaler & imputer pickled objects...")
        demo_imputer = Imputer(strategy="median")
        demo_scaler = StandardScaler()
        # fit imputer on train data
        demo_imputer.fit(train_df)
        # impute with train median value and scale train set
        train_df_imputed = demo_imputer.transform(train_df)
        demo_scaler.fit(train_df_imputed)

        # save scaler and imputer objects as pickled objects
        pickle.dump(demo_imputer, open("../../models/demo_train_imputer.pkl", "wb"))
        pickle.dump(demo_scaler, open("../../models/demo_train_scaler.pkl", "wb"))
    # else load objects from file
    else:
        print("Using supplied demographic scaler and imputer objects")
        # load scaler and imputer objects as pickled objects
        demo_imputer = pickle.load(open("../../models/demo_train_imputer.pkl", "rb"))
        demo_scaler = pickle.load(open("../../models/demo_train_scaler.pkl", "rb"))
        # impute with train median value and scale train set
        train_df_imputed = demo_imputer.transform(train_df)

    train_df_scaled = pd.DataFrame(demo_scaler.transform(train_df_imputed),
                                   index=train_df.index,
                                   columns=train_df.columns.values)
    # impute with train median value and scale using train mean/std dev
    val_df_imputed = demo_imputer.transform(val_df)
    val_df_scaled = pd.DataFrame(demo_scaler.transform(val_df_imputed),
                                 index=val_df.index,
                                 columns=val_df.columns.values)
    # impute with train median value and scale using train mean/std dev
    test_df_imputed = demo_imputer.transform(test_df)
    test_df_scaled = pd.DataFrame(demo_scaler.transform(test_df_imputed),
                                  index=test_df.index,
                                  columns=test_df.columns.values)

    demo_df = train_df_scaled.append(val_df_scaled)
    demo_df = demo_df.append(test_df_scaled)
    # drop duplicate rows
    demo_df = demo_df[~demo_df.index.duplicated(keep='first')]
    print(demo_df.head())
    print(demo_df.shape)
    return demo_df


def get_patient_from_file(path):
    """
    Function for getting the patient ID from a file where the file name contains the patient ID
    For MIMIC data, patient IDs take the format pXXXXXX-YYYY-MM-DD-HH-MM_ZZZZZ.ext, and we want to extract pXXXXXXX
    For UCI data, patient IDs take the format YYYY-MM-DD-XXXXXXXX_ZZZZZ.ext, and we want YYYY-MM-DD-XXXXXXXX

    :param str path: path to file
    :return: str patient ID
    """
    if project_configs.dataset == "mimic":
        return os.path.basename(path).split("-")[0]
    elif project_configs.dataset == "uci":
        return os.path.basename(path).split("_")[0]
    elif project_configs.dataset == "ucla":
        return "_".join(os.path.basename(path).split("_")[0:2])
    else:
        raise ValueError("only supported dataset options are mimic, uci, or ucla")


def get_window_from_file(path):
    """
    Function for getting the window number from a file where the file name contains the patient ID and window
    For MIMIC data, patient IDs take the format pXXXXXX-YYYY-MM-DD-HH-MM_ZZZZZ.ext, and we want to extract ZZZZZ
    For UCI data, patient IDs take the format YYYY-MM-DD-XXXXXXXX_ZZZZZ.ext, and we want ZZZZZ

    :param str path: path to file
    :return: str patient ID
    """
    if project_configs.dataset == "mimic":
        return int(os.path.basename(path).split("_")[1])
    elif project_configs.dataset == "uci":
        return int(os.path.basename(path).split("_")[1])
    elif project_configs.dataset == "ucla":
        return int(os.path.basename(path).split("_")[2])
    else:
        raise ValueError("only supported dataset options are mimic, uci, or ucla")


if __name__ == "__main__":
    import src.project_configs as project_configs
    # remove any empty files from train and test directories
    remove_empty_files(project_configs.train_dir)
    remove_empty_files(project_configs.val_dir)
    remove_empty_files(project_configs.test_dir)
    # split data into train/test
    train_val_test_split_mimic(data_dir="/Volumes/External/",
                               split_file_dir="data/",
                               create_file_list=False,
                               remove_old_dirs=False,
                               train_split=0.8,
                               val_split=0.1)
