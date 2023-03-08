#!/bin/bash

for i in 0 1 2 3 4 5 6 7 8 9
do 
    filename="lp_results_"$i
    python link_prediction.py --train_data data/FB15k-237/link-prediction/_train+valid_wiki-id.txt --test_data data/FB15k-237/link-prediction/_test_wiki-id.txt --entity_index data/FB15k-237/_wid2idx.json --rel_index data/FB15k-237/link-prediction/rel2idx.json --load_model fb15k237_rgcn_2_layers-basis_64-7_epochs.pt --graph data/FB15k-237/link-prediction/_train_wiki-id.txt --save_results $filename
done
