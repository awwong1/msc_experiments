#!/usr/bin/env bash

for i in {0..19}
do
    echo "Running Experiment $i";
    python3 -W ignore linear_beat_autoencoder.py --gpus 1 # --fast_dev_run True
    ckpt=`ls log_beat_autoencoder/version_${i}/checkpoints/*.ckpt`;
    python3 -W ignore sequence_autoencoder.py ${ckpt} --gpus 1 --batch_size 256 # --fast_dev_run True
    python3 -W ignore sequence_classifier.py ${ckpt} --gpus 1 --batch_size 256 # --fast_dev_run True
done
