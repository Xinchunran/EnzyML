#!/bin/bash  
##SBATCH --job-name=enzyme_baseline
##SBATCH --account=csb_gpu_acc
##SBATCH --partition=pascal
##SBATCH --nodes=1
##SBATCH --gres=gpu:2
##SBATCH --mem=40G
##SBATCH -t 0-02:00:00
##SBATCH --export=ALL

source activate ml

cd /home/ranx/workspace/enzy_basline

python machine_learning_basline.py enzyme_seqs_wo.fa enzyme_224_esm2/ values.txt
