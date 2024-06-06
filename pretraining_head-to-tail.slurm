#!/bin/bash
#SBATCH --job-name=CLIP_pretraining         # Job name
#SBATCH --mail-type=END,FAIL          # Mail events
#SBATCH --mail-user=andrea.papaluca@anu.edu.au     # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=16gb                    # Job memory request
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=CLIP_pretraining.log   # Standard output and error log
#SBATCH --time=120:00:00               # Time limit hrs:min:sec
pwd; hostname; date

#batchsize=512
batchsize=64
epochs=7
encoder='CompGCN'
dataset='FB15k-237-cut'
#dataset='WN18RR'
#dataset='YAGO3-10-cut'

echo "---- CLIP Pretraining ----"
cd ~/git/CLIP-KB/
source activate torch
echo $model_name
#srun python pretraining.py --train_data data/FB15k-237/link-prediction/train.txt --test_data data/FB15k-237/link-prediction/test.txt --entity_index data/FB15k-237/ent2idx.json --rel_index data/FB15k-237/rel2idx.json --head_to_tail --entities data/FB15k-237/pretraining/entities.json --save_model $model_name --batchsize 400
#srun python pretraining.py --train_data data/WN18RR/link-prediction/train_text.txt --test_data data/WN18RR/link-prediction/test_text.txt --entity_index data/WN18RR/ent2idx.json --rel_index data/WN18RR/rel2idx.json --head_to_tail --entities data/WN18RR/pretraining/entities.json --batchsize 128 --save_model $model_name

#srun python pretraining.py --dataset $dataset --graph_encoder $encoder --batchsize $batchsize --epochs $epochs --head_to_tail
#srun python pretraining.py --dataset $dataset --graph_encoder $encoder --batchsize $batchsize --epochs $epochs
srun python pretraining.py --dataset $dataset --graph_encoder $encoder --text_encoder "gpt2-xl" --batchsize $batchsize --epochs $epochs

echo "---- Done ----"
date
