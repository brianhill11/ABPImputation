import os
import glob
import sys
import pandas as pd

import abpimputation.project_configs as project_configs

sys.path.append("./")

# configuration for running pipeline
config = {
    "data_dir": "/data2/mimic/mimic_preprocessed",
    "output_dir": "/data2/mimic/mimic_preprocessed_imputedABP",
    "exec_dir": sys.executable,
}
print(config)

# get list of samples to process
# SAMPLES = [os.path.basename(x) for x in glob.glob(os.path.join(config["data_dir"], "*.csv.gz"))]
ids = pd.read_csv('validation_pIDs.txt', header=None).iloc[:, 0].tolist()
SAMPLES = []
for i in ids:
    SAMPLES.extend([os.path.basename(x) for x in glob.glob(os.path.join(config["data_dir"], i + "*"))])
print(SAMPLES)


rule all:
    input:
        expand("{output_dir}/{sample}",
               output_dir=config["output_dir"],
               sample=SAMPLES,
               )

rule impute:
    input:
        os.path.join(config["data_dir"], "{sample}")
    output:
        "{output_dir}/{sample}"
    shell:
        "{config[exec_dir]} driver.py "
        "--input-file {input} "
        "--save-dir {config[output_dir]} "

