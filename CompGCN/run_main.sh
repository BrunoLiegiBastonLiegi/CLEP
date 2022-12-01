#!/bin/bash

dataset=$1
n=$2
model=$3

for i in  $(seq 1 $n)
do
	python main.py --data $dataset --opn sub --batch 128 --epoch 100 --layer_size [200,200] --layer_dropout [0.3,0.3] --load_model $model --save_res "lp_results_cpp-compgcn_sub_"$dataset"_"$i".json" --entity_index ../data/$dataset/ent2idx.json --rel_index ../data/$dataset/rel2idx.json --num_workers 2
	python main.py --data $dataset --opn sub --batch 128 --epoch 100 --layer_size [200,200] --layer_dropout [0.3,0.3] --save_res "lp_results_baseline_"$dataset"_"$i".json" --entity_index ../data/$dataset/ent2idx.json --rel_index ../data/$dataset/rel2idx.json --num_workers 2
done
