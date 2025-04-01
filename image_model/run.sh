#!/bin/bash


#SBATCH --job-name tmc_img1
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 100gb
#SBATCH --time 24:00:00
#SBATCH --constraint interconnect_hdr
#SBATCH --output=/home/zeyut/tmc/image_model/job_output/%j.out

module purge

module load anaconda3/2023.09-0
source activate torch-1.13

cd /home/zeyut/tmc/image_model

python ./run.py 1 
