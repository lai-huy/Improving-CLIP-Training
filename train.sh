#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=sogclr_sgd           #Set the job name to "JobExample1"
#SBATCH --time=08:00:00                 #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                      #Request 1 task
#SBATCH --mem=8G                        #Request &G per node
#SBATCH --output=%j_deeplearning.log    #Send stdout/err to "Example1Out.[jobID]"

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python ./iSogCLR/bimodal_exps/clip.py \
    --data_path ./datasets \
    --ann_path ./clip_train \
    --train_file cc3m_train_subset.json \
    --train_image_root cc3m_subset_100k \
    --output_dir output/sogclr_cc3m_g0.8_e30 \
    --init_model \
    --use_amp \
    --ita_type sogclr \
    --tau_init 0.01 \
    --sogclr_gamma 0.8 \
    --eta_init 0.03 --sched cosine \
    --no-distributed \
    --epochs 30 \
    --opt sgd \
    --momentum 0.9
