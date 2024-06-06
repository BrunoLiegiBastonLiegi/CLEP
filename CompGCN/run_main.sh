#!/bin/bash

dataset=$1
n=$2
model=$3
model="../saved/models/$dataset/pretraining/CompGCN/$(basename $model)"
echo $model
epochs=1000
batchsize=1024
path="../saved/LP_results/"$dataset"/CompGCN/"
pret_file=$path"lp_results_"$dataset"_CompGCN-cpp_"$batchsize"bs_"$epochs"e.json"
base_file=$path"lp_results_"$dataset"_CompGCN-base_"$batchsize"bs_"$epochs"e.json"
ht=false

if [[ $model == *"h_to_t"* ]]
then
    ht=true
    path=$path"head_to_tail/"
    pret_file=$path"lp_results_"$dataset"_CompGCN-cpp_"$batchsize"bs_"$epochs"e_h_to_t.json"
fi

for i in  $(seq 1 $n)
do
    #python main.py --data $dataset --opn sub --batch $batchsize --epoch $epochs --layer_size [200,200] --layer_dropout [0.3,0.3] --load_model $model --entity_index ../data/$dataset/ent2idx.json --rel_index ../data/$dataset/rel2idx.json --num_workers 2
    #python main.py --data $dataset --opn sub --batch $batchsize --epoch $epochs --layer_size [200,200] --layer_dropout [0.3,0.3] --save_res "lp_results_baseline_"$dataset"_"$i".json" --entity_index ../data/$dataset/ent2idx.json --rel_index ../data/$dataset/rel2idx.json --num_workers 2
    python main.py --data $dataset --opn sub --batch $batchsize --epoch $epochs --layer_size [200,200] --layer_dropout [0.3,0.3] --load_model $model --num_workers 2 --save_res $pret_file --seed 42
    echo "------------------------------------------------------------------------------------------"
    echo "------------------------------------------------------------------------------------------"
    python main.py --data $dataset --opn sub --batch $batchsize --epoch $epochs --layer_size [200,200] --layer_dropout [0.3,0.3] --num_workers 2 --save_res $base_file --seed 42
    outfile=$path"lp_results_"$dataset"_CompGCN_"$batchsize"bs_"$epochs"e_run-"$i".json"
    if $ht
    then
	outfile=$path"lp_results_"$dataset"_CompGCN_"$batchsize"bs_"$epochs"e_h_to_t_run-"$i".json"
    fi
    python merge_res.py $pret_file $base_file $outfile
done
