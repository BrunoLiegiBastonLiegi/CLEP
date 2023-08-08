
# Latent space visualization
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
import colorcet as cc
import json
import bitsandbytes as bnb

def visualize_embeddings(embeddings, n_clusters=None, clusters=None, ax=plt.subplots()[1]):
    if n_clusters !=None:
        if n_clusters == 'auto':
            sse = {}
            for n in range(2,200):
                print(f'k-means {n}/200', end='\r')
                kmeans = KMeans(n_clusters=n, random_state=0).fit(embeddings)
                sse[n] = kmeans.inertia_
            plt.plot(list(sse.keys()), list(sse.values()))
            plt.show()
            n_clusters = input('Number of clusters: ')
        #clusters = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0).fit_predict(embeddings)
        clusters = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(embeddings)
    elif clusters == None:
        clusters = [1 for i in range(len(embeddings))]
    proj = TSNE(n_components=2, init='pca').fit_transform(embeddings)
    ax.scatter(proj[:,0], proj[:,1], c=clusters, cmap=cc.cm.glasbey)
    return clusters

# Training
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

def training_routine(model, step_f, train_data, test_data, epochs, batchsize, learning_rate, valid_data=None, eval_f=None, eval_each=-1, unfreezing_f=None, accum_iter=1, dev=torch.device('cpu')):
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.)
    scaler = GradScaler()

    train_loader = DataLoader(
        train_data,
        batch_size = batchsize,
        shuffle = True,
        collate_fn = train_data.collate_fn
    )

    test_loader = DataLoader(
        test_data,
        batch_size = batchsize,
        shuffle = True,
        collate_fn = test_data.collate_fn
    )
    
    if valid_data is None:
        valid_loader = test_loader
    else:
        valid_loader = DataLoader(
            valid_data,
            batch_size = batchsize,
            shuffle = True,
            collate_fn = test_data.collate_fn
        )

    train_loss, valid_loss, metrics = [], [], {}
    print_steps = int(len(train_loader)/5)
    for e in range(epochs):
        if unfreezing_f is not None:
            unfreezing_f(model, e)
        print(f'\n### EPOCH {e}')
        running_loss, epoch_loss = 0., 0.
        model.train()
        for i, (batch, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            with autocast():
                #batch, label = batch.to(dev), label.to(dev)
                loss = step_f(model, batch, label, dev)
                running_loss += loss.item()
                # normalize loss to account for batch accumulation
                loss = loss / accum_iter 
            scaler.scale(loss).backward()
            # weights update
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if i % print_steps == print_steps - 1:
                epoch_loss += running_loss
                print(f'[{e}, {i*batchsize}]\t Loss: {running_loss/print_steps:.4f}') # The average is not correct if len(train_loader) % batchsize != 0
                running_loss = 0.
        running_loss = 0.
        model.eval()
        for i, (batch, label) in enumerate(valid_loader):
            with torch.no_grad():
                loss = step_f(model, batch, label, dev)
                running_loss += loss.item()
        valid_loss.append(running_loss/(len(valid_loader)))
        train_loss.append(epoch_loss/len(train_loader))
        print(f'> Valid Loss: {running_loss/(len(valid_loader)):.4f}')
        if e % eval_each == eval_each -1 and eval_f != None: # run evaluation every eval_each epochs
            with torch.no_grad():
                metrics['Epoch '+str(e)] = eval_f(model, valid_data) if valid_data is not None else eval_f(model, test_data)
            print(f'### Evaluation Metrics after {e+1} epochs:')
            print(json.dumps(metrics['Epoch '+str(e)], indent=2))
    return train_loss, valid_loss, metrics

# Graph
from dgl import graph, heterograph
import sys
sys.path.append('CompGCN')
from compgcn_utils import in_out_norm
class KG(object):
    """
    Simple wrapper to the dgl graph object.
    """
    def __init__(self, embedding_dim, ent2idx, rel2idx, triples=None, dev=torch.device('cpu'), add_inverse_edges=False):
        super().__init__()
        self.dev = dev
        self.emb_dim = embedding_dim
        self.add_inverse_edges = add_inverse_edges
        self.r2idx = rel2idx
        self.e2idx = ent2idx
        if triples != None:
            if add_inverse_edges:
                inv_triples = triples[:,[2,1,0]]
                inv_triples[:,1] += len(rel2idx)
                triples = torch.vstack((triples, inv_triples))
            self.g = graph((triples[:,0], triples[:,2]), num_nodes=len(ent2idx), device=self.dev)
            self.etypes = triples[:,1].to(self.dev)
            self.g.edata['etype'] = self.etypes
            #self.node_feat = torch.nn.Embedding(self.g.num_nodes(), self.emb_dim).to(self.dev) # random initial node features
            # identify in and out edges
            in_edges_mask = [True] * (self.g.num_edges() // 2) + [False] * (
                self.g.num_edges() // 2
            )
            out_edges_mask = [False] * (self.g.num_edges() // 2) + [True] * (
                self.g.num_edges() // 2
            )
            self.g.edata["in_edges_mask"] = torch.Tensor(in_edges_mask).to(dev)
            self.g.edata["out_edges_mask"] = torch.Tensor(out_edges_mask).to(dev)
            self.g = in_out_norm(self.g)
                
    def build_from_file(self, infile):
        triples = []
        with open(infile, 'r') as f:
            for l in f:
                head, rel, tail = l.split()
                head, rel, tail = self.e2idx[head], self.r2idx[rel], self.e2idx[tail]
                triples.append([head, rel, tail])
                if self.add_inverse_edges:
                    triples.append([tail, rel+len(self.r2idx), head])
        triples = torch.as_tensor(triples)
        self.g = graph((triples[:,0], triples[:,2]), num_nodes=len(self.e2idx), device=self.dev)
        self.etypes = triples[:,1].to(self.dev)
        self.g.edata['etype'] = self.etypes
        #self.node_feat = torch.nn.Embedding(self.g.num_nodes(), self.emb_dim).to(self.dev) if node_features == None else node_features# random initial node features
        # identify in and out edges
        in_edges_mask = [True] * (self.g.num_edges() // 2) + [False] * (
            self.g.num_edges() // 2
        )
        out_edges_mask = [False] * (self.g.num_edges() // 2) + [True] * (
            self.g.num_edges() // 2
        )
        self.g.edata["in_edges_mask"] = torch.Tensor(in_edges_mask).to(self.dev)
        self.g.edata["out_edges_mask"] = torch.Tensor(out_edges_mask).to(self.dev)
        self.g = in_out_norm(self.g)

    @property
    def n_rel(self):
        return len(set(self.etypes.tolist()))

    @property
    def embedding_dim(self):
        return self.emb_dim


class SimilarityQA(object):

    def __init__(self, clep_model, tokenizer):
        super(self, ).__init__()
        self.clep = clep_model
        self.tok = tokenizer

    def answer(self, query, top_k=10):
        query = self.tok(query)
        query_emb = torch.nn.functional.normalize(self.clep.t_nn(query, p=2, dim=-1))
        node_embs = torch.nn.functional.normalize(self.clep.g_nn(:))
        similarities = (query_emb*node_embs).sum(-1)
        values, indices = similarities.sort(descending=True)
        return list(zip(indices[:top_k], values[:top_k]))
                                                  
        
