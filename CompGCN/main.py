import argparse, json, os
from time import time

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_loader import Data
from compgcn import CompGCN_ConvE
from compgcn_utils import in_out_norm

import dgl.function as fn

import sys, json, torch
sys.path.append('../')
from model import CLIP_KB, GPT2CaptionEncoder, CompGCNWrapper
from utils import KG
from dataset import LinkPredictionDataset
from os.path import basename


# predict the tail for (head, rel, -1) or head for (-1, rel, tail)
def predict(model, graph, device, data_iter, split="valid", mode="tail"):
    model.eval()
    with th.no_grad():
        results = {}
        train_iter = iter(data_iter["{}_{}".format(split, mode)])

        for step, batch in enumerate(train_iter):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label,
            )
            pred = model(graph, sub, rel)
            b_range = th.arange(pred.size()[0], device=device)
            target_pred = pred[b_range, obj]
            pred = th.where(label.byte(), -th.ones_like(pred) * 10000000, pred)
            pred[b_range, obj] = target_pred

            # compute metrics
            ranks = (
                1
                + th.argsort(
                    th.argsort(pred, dim=1, descending=True),
                    dim=1,
                    descending=False,
                )[b_range, obj]
            )
            ranks = ranks.float()
            results["ranks"] = ranks.tolist() + results.get("ranks", [])
            results["count"] = th.numel(ranks) + results.get("count", 0.0)
            results["mr"] = th.sum(ranks).item() + results.get("mr", 0.0)
            results["mrr"] = th.sum(1.0 / ranks).item() + results.get(
                "mrr", 0.0
            )
            for k in [1, 3, 10]:
                results["hits@{}".format(k)] = th.numel(
                    ranks[ranks <= (k)]
                ) + results.get("hits@{}".format(k), 0.0)
    return results


# evaluation function, evaluate the head and tail prediction and then combine the results
def evaluate(model, graph, device, data_iter, split="valid"):
    # predict for head and tail
    left_results = predict(model, graph, device, data_iter, split, mode="tail")
    right_results = predict(model, graph, device, data_iter, split, mode="head")
    results = {}
    count = float(left_results["count"])

    # combine the head and tail prediction results
    # Metrics: MRR, MR, and Hit@k
    results["left_ranks"] = left_results["ranks"]
    results["right_ranks"] = right_results["ranks"]
    results["left_mr"] = round(left_results["mr"] / count, 5)
    results["left_mrr"] = round(left_results["mrr"] / count, 5)
    results["right_mr"] = round(right_results["mr"] / count, 5)
    results["right_mrr"] = round(right_results["mrr"] / count, 5)
    results["mr"] = round(
        (left_results["mr"] + right_results["mr"]) / (2 * count), 5
    )
    results["mrr"] = round(
        (left_results["mrr"] + right_results["mrr"]) / (2 * count), 5
    )
    for k in [1, 3, 10]:
        results["left_hits@{}".format(k)] = round(
            left_results["hits@{}".format(k)] / count, 5
        )
        results["right_hits@{}".format(k)] = round(
            right_results["hits@{}".format(k)] / count, 5
        )
        results["hits@{}".format(k)] = round(
            (
                left_results["hits@{}".format(k)]
                + right_results["hits@{}".format(k)]
            )
            / (2 * count),
            5,
        )
    return results


def main(args):

    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    # check cuda
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    with open('../data/{}/ent2idx.json'.format(args.dataset), 'r') as f:
        ent2idx = json.load(f)
    with open('../data/{}/rel2idx.json'.format(args.dataset), 'r') as f:
        rel2idx = json.load(f)

    # construct graph, split in/out edges and prepare train/validation/test data_loader
    data = Data(
        args.dataset, args.lbl_smooth, args.num_workers, args.batch_size, ent2idx=ent2idx, rel2idx=rel2idx
    )
    data_iter = data.data_iter  # train/validation/test data_loader
    graph = data.g.to(device)
    num_rel = th.max(graph.edata["etype"]).item() + 1

    # Compute in/out edge norms and store in edata
    graph = in_out_norm(graph)

    if args.load_model != None:
        # all this just to load a pretrained compgcn is really annoying
        kg = KG(embedding_dim=200, ent2idx=ent2idx, rel2idx=rel2idx, dev=device, add_inverse_edges=True)
        kg.build_from_file('./' + args.dataset + '/train.txt')
        graph = kg.g
        _ = GPT2CaptionEncoder(pretrained_model='gpt2-xl')
        conf = {
            'kg': kg,
            'n_layers': 2,
            'indim': kg.embedding_dim,
            'hdim': 200,
            'num_bases': -1,
            'comp_fn' : 'sub',
            'return_rel_embs':  True
        }
        compgcn = CompGCNWrapper(**conf)
        clip = CLIP_KB(
            graph_encoder = compgcn,
            text_encoder = _,
            hdim = 200
        ).to(device)
        clip.load_state_dict(torch.load(args.load_model))
        compgcn = clip.g_encoder.model
        mapping_net = clip.g_mlp
    else:
        compgcn = None
        mapping_net = None

    # Step 2: Create model =================================================================== #
    compgcn_model = CompGCN_ConvE(
        num_bases=args.num_bases,
        num_rel=num_rel,
        num_ent=graph.num_nodes(),
        in_dim=args.init_dim,
        layer_size=args.layer_size,
        comp_fn=args.opn,
        batchnorm=True,
        dropout=args.dropout,
        layer_dropout=args.layer_dropout,
        num_filt=args.num_filt,
        hid_drop=args.hid_drop,
        feat_drop=args.feat_drop,
        ker_sz=args.ker_sz,
        k_w=args.k_w,
        k_h=args.k_h,
        compgcn=compgcn,
        mapping_net=mapping_net
    )
    compgcn_model = compgcn_model.to(device)

    # Step 3: Create training components ===================================================== #
    loss_fn = th.nn.BCELoss()
    optimizer = optim.Adam(
        compgcn_model.parameters(), lr=args.lr, weight_decay=args.l2
    )

    # Step 4: training epoches =============================================================== #
    best_mrr = 0.0
    kill_cnt = 0
    results = {}
    saved_model_name = "../saved/models/{}/link-prediction/CompGCN/comp_link_".format(args.dataset)
    if args.load_model is None:
        saved_model_name += "Baseline"
    else:
        tmp = '_Finetuned_from_{}'.format(basename(args.load_model))
        saved_model_name += tmp
    saved_model_name += "_{}_{}bs_{}e".format(args.dataset, args.batch_size, args.max_epochs)
    os.makedirs(os.path.dirname(saved_model_name), exist_ok=True)
    for epoch in range(args.max_epochs):
        # Training and validation using a full graph
        compgcn_model.train()
        train_loss = []
        t0 = time()
        for step, batch in enumerate(data_iter["train"]):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label,
            )
            logits = compgcn_model(graph, sub, rel)

            # compute loss
            tr_loss = loss_fn(logits, label)
            train_loss.append(tr_loss.item())

            # backward
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

        train_loss = np.sum(train_loss)

        t1 = time()
        val_results = evaluate(
            compgcn_model, graph, device, data_iter, split="valid"
        )
        val_results.pop('left_ranks')
        val_results.pop('right_ranks')
        t2 = time()
        results[epoch] = val_results

        # validate
        if val_results["mrr"] > best_mrr:
            best_mrr = val_results["mrr"]
            best_epoch = epoch
            th.save(
                compgcn_model.state_dict(), saved_model_name
            )
            kill_cnt = 0
            print("saving model...")
        else:
            kill_cnt += 1
            if kill_cnt > 100:
                print("early stop.")
                break
        print(
            "In epoch {}, Train Loss: {:.4f}, Valid MRR: {:.5}\n, Train time: {}, Valid time: {}".format(
                epoch, train_loss, val_results["mrr"], t1 - t0, t2 - t1
            )
        )

    # test use the best model
    compgcn_model.eval()
    compgcn_model.load_state_dict(th.load(saved_model_name))
    test_results = evaluate(
        compgcn_model, graph, device, data_iter, split="test"
    )
    results['test'] = test_results
    print(
        "Test MRR: {:.5}\n, MR: {:.10}\n, H@10: {:.5}\n, H@3: {:.5}\n, H@1: {:.5}\n".format(
            test_results["mrr"],
            test_results["mr"],
            test_results["hits@10"],
            test_results["hits@3"],
            test_results["hits@1"],
        )
    )
    if args.save_res is None:
        args.save_res = "../saved/LP_results/{}/CompGCN/lp_results_".format(args.dataset) + basename(saved_model_name).replace('comp_link', '') + '.json'
    print(f'Saving results to: {args.save_res}')
    os.makedirs(os.path.dirname(args.save_res), exist_ok=True)
    with open(args.save_res, 'w') as f:
        json.dump(results, f, indent=2)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        dest="dataset",
        default="FB15k-237",
        help="Dataset to use, default: FB15k-237",
    )
    parser.add_argument(
        "--model", dest="model", default="compgcn", help="Model Name"
    )
    parser.add_argument(
        "--score_func",
        dest="score_func",
        default="conve",
        help="Score Function for Link prediction",
    )
    parser.add_argument(
        "--opn",
        dest="opn",
        default="sub",
        help="Composition Operation to be used in CompGCN",
    )

    parser.add_argument(
        "--batch", dest="batch_size", default=1024, type=int, help="Batch size"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default="0",
        help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0",
    )
    parser.add_argument(
        "--epoch",
        dest="max_epochs",
        type=int,
        default=500,
        help="Number of epochs",
    )
    parser.add_argument(
        "--l2", type=float, default=0.0, help="L2 Regularization for Optimizer"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Starting Learning Rate"
    )
    parser.add_argument(
        "--lbl_smooth",
        dest="lbl_smooth",
        type=float,
        default=0.1,
        help="Label Smoothing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of processes to construct batches",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        default=41504,
        type=int,
        help="Seed for randomization",
    )

    parser.add_argument(
        "--num_bases",
        dest="num_bases",
        default=-1,
        type=int,
        help="Number of basis relation vectors to use",
    )
    parser.add_argument(
        "--init_dim",
        dest="init_dim",
        default=200,
        type=int,
        help="Initial dimension size for entities and relations",
    )
    parser.add_argument(
        "--layer_size",
        nargs="?",
        default="[200,200]",
        help="List of output size for each compGCN layer",
    )
    parser.add_argument(
        "--gcn_drop",
        dest="dropout",
        default=0.1,
        type=float,
        help="Dropout to use in GCN Layer",
    )
    parser.add_argument(
        "--layer_dropout",
        nargs="?",
        default="[0.3]",
        help="List of dropout value after each compGCN layer",
    )

    # ConvE specific hyperparameters
    parser.add_argument(
        "--hid_drop",
        dest="hid_drop",
        default=0.3,
        type=float,
        help="ConvE: Hidden dropout",
    )
    parser.add_argument(
        "--feat_drop",
        dest="feat_drop",
        default=0.3,
        type=float,
        help="ConvE: Feature Dropout",
    )
    parser.add_argument(
        "--k_w", dest="k_w", default=10, type=int, help="ConvE: k_w"
    )
    parser.add_argument(
        "--k_h", dest="k_h", default=20, type=int, help="ConvE: k_h"
    )
    parser.add_argument(
        "--num_filt",
        dest="num_filt",
        default=200,
        type=int,
        help="ConvE: Number of filters in convolution",
    )
    parser.add_argument(
        "--ker_sz",
        dest="ker_sz",
        default=7,
        type=int,
        help="ConvE: Kernel size to use",
    )
    parser.add_argument(
        "--load_model",
        default=None,
        help='Path to model to load.'
    )
    parser.add_argument('--save_res')
    #parser.add_argument('--entity_index')
    #parser.add_argument('--rel_index')

    args = parser.parse_args()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    print(args)

    args.layer_size = eval(args.layer_size)
    args.layer_dropout = eval(args.layer_dropout)

    main(args)
