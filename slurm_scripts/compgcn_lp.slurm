#!/bin/bash
#SBATCH --job-name=compgcn_lp         # Job name
#SBATCH --mail-type=END,FAIL          # Mail events
#SBATCH --mail-user=andrea.papaluca@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=16gb                    # Job memory request
#SBATCH --partition=bigdata
#SBATCH --gres=gpu:1
#SBATCH --qos=bigdata
#SBATCH --output=compgcn_lp.log   # Standard output and error log
#SBATCH --time=1680:00:00               # Time limit hrs:min:sec
pwd; hostname; date

n=1
#dataset=WN18RR
dataset=FB15k-237-cut
#dataset="YAGO3-10-cut"
#model="../saved/models/"$dataset"/pretraining/CompGCN/"$(basename $1)
model="../"$(basename $1)
echo $model

echo "---- Link Prediction ----"
cd ~/git/CLIP-KB/CompGCN/
source activate torch
srun bash run_main.sh $dataset $n $model
echo "---- Done ----"
date
