#!/bin/bash
#
#################
#set a job name
#SBATCH --job-name=lstm_gpu
#################
#a file for job output, you can check job progress
#SBATCH --output=lstm_gpu.out
#################
# a file for errors from the job
#SBATCH --error=lstm_gpu.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=24:00:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal # high_p
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, owners, hns, bigmem (jobs requiring >64Gigs RAM)
#
#############
#number of nodes you are requesting
#################
#SBATCH --gpus 1
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
# --mail-user=ramya.rangan117@gmail.com
# --mail-type=ALL

#now run normal batch commands
# note the "CMD BATCH is an R specific command
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --partition=gpu

source ~/.bashrc

module load python/3.6.1
module load py-tensorflow/2.4.1_py36

source /scratch/users/ramyar/cs230/cs230_project/code/cs230_venv/bin/activate

python lstm_model.py ../data/train_dev_test/train.csv ../data/train_dev_test/dev.csv

deactivate
