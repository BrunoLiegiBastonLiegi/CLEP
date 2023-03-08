# CLEP: Constrastive Language-Entity Pretraining

Implementation of what presented in the paper [...](...)

## Citation

If you find our work useful please consider citing us:

```
.
.
.
```

## Requirements

The three main libraries used to build the experiments are the following:

- pytorch
- dgl
- transformers

Several other common python libraries are needed as well, such as: `numpy`, `matplotlib`, `scipy`, `tqdm` 

## Dataset Preparation

The dataset directories are all located under `data/`. To prepare one of them for running you just need to copy the original `train.txt`, `valid.txt` and `test.txt` triples files in the proper dataset directory `data/*dataset*/link-prediction/`. For example, for `FB15k-237`:

```
cp path_to_FB15k-237/*.txt data/FB15k-237/link-prediction/
```

Then, move to the `data/` directory if you haven't yet (`cd data/`) and run the generation script `gen.py` by specifying the dataset:

```
python gen.py FB15k-237/
```

This is going to generate the files needed for the entity-description pretraining under `data/FB15k-237/pretraining/` and the new triples files under `data/FB15k-237/link-prediction` (after creating a backup of the original ones).

In case you wish to remove the nodes whose description is missing just pass the `--cut` argument:

```
python gen.py FB15k-237/ --cut
```

A new `data/FB15k-237-cut/` directory is going to be created (if it didn't exist yet) with the same structure as just described.

Now you are ready for running the experiments!

## Contrastive Pretraining

In order to run the entity-description matching pretraining on a dataset, e.g. the cut `FB15k-237` dataset, just give:

```
python pretraining.py --dataset FB15k-237-cut
```

Several other options can be passed to this script as well, but most notably:
```
--load_model: Pass a pretrained model, skip training loop and procede directly to evaluation.
--batchsize: Define the size of the batches.
--epochs: Number of training epochs.
--head_to_tail: Match head entities with tail descriptions.
--graph_encoder: Specify whether to use either `RGCN` or `CompGCN`.
--save_model: Save the model to a path different from the default one (`saved/models/pretraining/*dataset*/*graph_encoder*/`).
```

To perform link prediction using entity-description similarity as illustrated in the paper just pass the `--head_to_tail` option.
## Link Prediction

In order to run a link prediction experiment you can do the following.

If you wish to train an RGCN you can use the `link_prediction.py` script and specify the dataset:

```
python link_prediction.py --dataset FB15k-237-cut
```

additionally you can specify:

```
--load_model: Load an RGCN pretrained on the entity-description matching.
--batchsize: Define the size of the batches.
--epochs: Number of training epochs.
--LP_head: Specify whether to use `Distmult` or `ConvE`.
--one_to_N_scoring: Use the 1-N triple scoring method.
--save_results: Save the link prediction metrics to a path different from the default one (saved/LP_results/*dataset*/RGCN/).
```

For the CompGCN instead, we relied on an unofficial dgl [implementation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/compGCN) that we adapted fo our use case. All the components needed to run the CompGCN examples are located under the `CompGCN/` directory.

In order to train a CompGCN, firstly move to the `CompGCN/` directory with `cd CompGCN/` and then run:

```
python main.py --data FB15k-237-cut
```

again, other options can be specified, such as:

```
--epoch: Maximum number of epochs.
--batch: Size of the batches.
--load_model: Load a CompGCN pretrained on the entity-description matching.
--save_res: Save the link prediction metrics to a path different from the default one (saved/LP_results/*dataset*/CompGCN/).
```