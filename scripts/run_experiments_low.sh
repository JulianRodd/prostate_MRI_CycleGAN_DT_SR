#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --qos=low
#SBATCH --job-name="run_experiments"
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --output=./output_files/slurm-fbgb-%j.out
#SBATCH --container-mounts=/data/temporary/julian:/data/temporary/julian
#SBATCH --container-image="dodrio1.umcn.nl#uokbaseimage/diag:latest"

cd /data/temporary/julian/scripts/prostate-SR_domain-correction/domain_cor/prostate-SR_domain-correction
pip3 install -r requirements.txt
cd /data/temporary/julian/scripts/prostate-SR_domain-correction/domain_cor/prostate-SR_domain-correction/commands
git fetch origin
git reset --hard origin/main
tree .
bash run.sh train-server sweep_result_best_params_small