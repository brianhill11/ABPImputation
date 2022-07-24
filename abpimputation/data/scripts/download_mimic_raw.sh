#!/bin/sh

record_file='../metadata/ECG_PLETH_ABP_IDs_wdb3_matched.txt'
save_dir='../raw/'

# first argument is record file
if [ $# -eq 1 ]
    then
        record_file=$1
fi

# second argument is save dir
if [ $# -eq 2 ]
    then 
        record_file=$1
        save_dir=$2
fi
echo "Using record file: ${record_file}"
echo "Saving records to: ${save_dir}"

# for each record, download if it is not already downloaded
for i in `sort -R ${record_file}`;
do
    echo $(dirname ${i})/
    wget --directory-prefix=${save_dir} -r -np -nc https://physionet.org/files/mimic3wdb-matched/1.0/$(dirname ${i})/
done
