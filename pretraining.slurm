#!/bin/bash
#SBATCH --job-name=CLIP_pretraining         # Job name
#SBATCH --mail-type=END,FAIL          # Mail events
#SBATCH --mail-user=andrea.papaluca@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=16gb                    # Job memory request
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=CLIP_pretraining.log   # Standard output and error log
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
pwd; hostname; date

batchsize=512
epochs=3
#encoder='CompGCN'
encoder='RGCN'
#dataset='WN18RR'
#dataset='FB15k-237'
#dataset='FB15k-237-cut'
#dataset='YAGO3-10-cut'
dataset="wikidata-disambig"
#text_encoder='distilbert-base-cased'
text_encoder='gpt2'

echo "---- CLIP Pretraining ----"
echo $SHELL
source activate nlp
pip list | grep torch
cd ~/git/CLIP-KB/
python3.8 pretraining.py --dataset $dataset --graph_encoder $encoder --batchsize $batchsize --epochs $epochs --text_encoder $text_encoder
echo "---- Done ----"
date
