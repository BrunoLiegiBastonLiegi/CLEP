#!/bin/bash

n=5
epochs=1000
#epochs=1

for i in $(seq 3 $n)
do
	#python main.py --score_func conve --opn sub --gpu 1 --data FB15k-237 --load_model ../compgcn_2-layers_sub_8-epochs_fb15k237.pt --epoch $epochs --save_res 'results_Caption_Pretraining_'$i'.json'
        python main.py --score_func conve --opn sub --gpu 0 --data FB15k-237 --layer_size [200,200] --layer_dropout [0.3,0.3] --epoch $epochs --save_res 'results_Baseline_'$i'.json'
done
