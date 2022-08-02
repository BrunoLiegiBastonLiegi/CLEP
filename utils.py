
# Latent space visualization
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np
import colorcet as cc

def visualize_embeddings(embeddings, n_clusters=None, ax=plt.subplots()[1]):
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
        clusters = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0).fit_predict(embeddings)
    else:
        clusters = [1 for i in range(len(embeddings))]
    proj = TSNE(n_components=2,init='pca').fit_transform(embeddings)
    ax.scatter(proj[:,0], proj[:,1], c=clusters, cmap=cc.cm.glasbey)
    return clusters

# Training
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

def training_routine(model, step_f, train_data, test_data, epochs, batchsize, accum_iter=1, dev=torch.device('cpu')):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
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

    train_loss, test_loss = [], []
    print_steps = int(len(train_loader)/5)
    for e in range(epochs):
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
        for i, (batch, label) in enumerate(test_loader):
            with torch.no_grad():
                loss = step_f(model, batch, label, dev)
                running_loss += loss.item()
        test_loss.append(running_loss/(len(test_loader)))
        train_loss.append(epoch_loss/len(train_loader))
        print(f'> Test Loss: {running_loss/(len(test_loader)):.4f}')
    return train_loss, test_loss

# Graph
from dgl import graph
class KG(object):
    def __init__(self, triples=None):
        super().__init__()
        
    def build_from_file(self, infile, ent2idx, rel2idx):
        triples, missing = [], {}
        with open(infile, 'r') as f:
            for l in f:
                t = l.split()
                try:
                    head = ent2idx[t[0]]
                except:
                    missing.update({t[0]:0})
                try:
                    rel = rel2idx[t[1]]
                except:
                    missing.update({t[1]:0})
                try:
                    tail = ent2idx[t[2]]
                except:
                    missing.update({t[2]:0})
                try:
                    triples.append([head,rel,tail])
                except:
                    continue
        triples = torch.as_tensor(triples)
        self.g = graph((triples[:.0], triples[:,2]))
        self.etypes = triples[:,1]
        self.node_feat = torch.randn(self.g.num_nodes, 200)
        #print(f'> {len(missing)} missing mappings.')
