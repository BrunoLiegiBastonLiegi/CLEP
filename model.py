import torch
import numpy as np
from torch.nn import Linear, BatchNorm1d, Dropout, ReLU, Sequential
from torch.nn.functional import normalize
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, BertModel
from dgl.nn import RelGraphConv

import sys
sys.path.insert(1, 'CompGCN')
from compgcn import CompGraphConv
import compgcn


class MLP(torch.nn.Module):

    def __init__(self, n_layers: int, indim: int, hdim: int, outdim: int = -1, activation = ReLU(), normalization = Dropout(0.1)):
        super().__init__()
        self.n_layers = n_layers
        self.indim = indim
        self.hdim = hdim
        layers = [Linear(indim, hdim)]
        for n in range(n_layers-1):
            layers.append(normalization)
            layers.append(activation)
            if n == n_layers-1 and outdim != -1:
                layers.append(Linear(hdim, outdim))
            else:
                layers.append(Linear(hdim, hdim))
        self.nn = Sequential(*layers)
        del layers

    def forward(self, x):
        return self.nn(x)

class TransformerEncoder(torch.nn.Module):

    def __init__(self, n_layers: int, indim: int, hdim: int, nhead: int = 4):
        super().__init__()
        self.n_layers = n_layers
        self.indim = indim
        self.hdim = hdim
        self.layer = torch.nn.TransformerEncoderLayer(
            d_model = indim,
            nhead = nhead,
            dim_feedforward = indim,
            batch_first = True
        )
        self.nn = Sequential(
            torch.nn.TransformerEncoder(self.layer, n_layers),
            Linear(indim, hdim)
            )

    def forward(self, x):
        return self.nn(x)

class CLIP_KB(torch.nn.Module):

    def __init__(self, graph_encoder, text_encoder, hdim: int):
        super().__init__()
        self.hdim = hdim
        # Temperature
        #self.register_parameter(
        #    name = 'T',
        #    param = torch.nn.parameter.Parameter(torch.tensor(0.07), requires_grad = True)
        #)
        self.register_parameter(
            name = 'T',
            param = torch.nn.parameter.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad = True)
        )
        # text encoding
        self.t_encoder = text_encoder
        self.t_mlp = MLP(1, self.t_encoder.hdim, hdim) # Switch dropout with BatchNorm!?
        #self.t_mlp = TransformerEncoder(1, self.t_encoder.hdim, hdim)
        self.t_nn = Sequential(self.t_encoder, self.t_mlp)
        # graph encoding
        self.g_encoder = graph_encoder
        self.g_mlp = MLP(1, self.g_encoder.hdim, hdim)
        #self.g_mlp = TransformerEncoder(1, self.g_encoder.hdim, hdim)
        self.g_nn = Sequential(self.g_encoder, self.g_mlp)

    def forward(self, nodes, captions):
        self.T = min(self.T, 100)
        #return ( normalize(self.g_mlp(self.g_encoder(nodes)), p=2, dim=-1),
        #         normalize(self.t_mlp(self.t_encoder(captions)), p=2, dim=-1) )
        return ( normalize(self.g_nn(nodes), p=2, dim=-1),
                 normalize(self.t_nn(captions), p=2, dim=-1) )

class PretrainedGraphEncoder(torch.nn.Module):

    def __init__(self, node_embeddings: dict, index: dict, device: torch.device):
        super().__init__()
        #self.node2emb = node_embeddings
        self.dev = device
        self._hdim = list(node_embeddings.values())[0].shape[-1]
        self.ordered_embs = torch.zeros(len(index), self._hdim, dtype=float)
        embs = {}
        n = 0
        for k,v in index.items():
            try:
                e = torch.as_tensor(node_embeddings[k])
                embs[k] = e
                self.ordered_embs[v] = e
            except:
                n += 1
                embs[k] = torch.zeros(self._hdim)
        print(f'Warning: {n} pretrained embeddings were missing. They were substituted with zeros.')
        # need to explictly cast to float32
        #self.ordered_embs = torch.as_tensor(np.vstack(list(embs.values())), dtype=torch.float)
        del embs
        #self.register_parameter(
        #    name='ordered_embs',
        #    param=torch.nn.Parameter(
        #        torch.as_tensor(np.vstack(list(node_embeddings.values()))),
        #        requires_grad = False
        #    )
        #)
        #self.register_parameter(
        #    name='ordered_embs',
        #    param=torch.nn.Parameter(
        #        torch.as_tensor(list(zip(
        #            *sorted(node_embeddings.items(),
        #                    key=lambda x: x[0])
        #        ))[1]),
        #        requires_grad = False
        #    )
        #)
        #self.ordered_embs = torch.vstack(list(zip(
        #    *sorted(node_embeddings.items(),
        #            key=lambda x: x[0])
        #))[1])

    def forward(self, nodes):
        return self.ordered_embs[nodes].squeeze(1).to(self.dev) # dinamically move to device the batch
        #return self.ordered_embs[nodes].squeeze(1)
    
    @property
    def hdim(self):
        #return self.ordered_embs.shape[-1]
        return self._hdim
        
class GPT2CaptionEncoder(torch.nn.Module):

    def __init__(self, pretrained_model: str = 'gpt2'):
        super().__init__()
        #self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.model = GPT2Model.from_pretrained(pretrained_model)       # which one is better to use?
        for m in self.model.h[:-4]: # freezing every layer but the last 4
            for p in m.parameters():
                p.requires_grad = False
        
    def forward(self, x):
        #return self.model(**x).logits[:,-1,:]
        return self.model(**x).last_hidden_state[:,-1,:]

    @property
    def hdim(self):
        #return self.model.config.vocab_size
        return self.model.config.n_embd

class BertCaptionEncoder(torch.nn.Module):

    def __init__(self, pretrained_model: str = 'bert-base-cased'):
        super().__init__()
        self.model = BertModel.from_pretrained(pretrained_model)       
        for m in self.model.encoder.layer[:-4]: # freezing every layer but the last 4
            for p in m.parameters():
                p.requires_grad = False
        
    def forward(self, x):
        return self.model(**x).last_hidden_state[:,0,:]

    @property
    def hdim(self):
        return self.model.embeddings.word_embeddings.embedding_dim

    #def unfreeze_layer(self, i: int):
     #   for p in self.model.


class LinkPredictionModel(torch.nn.Module):

    def __init__(self, graph_embedding_model, rel2idx : dict, mode: str = 'Distmult', external_rel_embs=False):
        super().__init__()
        assert mode in ('Distmult', 'TransE', 'Rescal', 'ConvE')
        self.mode = mode
        self.external_rel_embs = external_rel_embs
        self.model = graph_embedding_model
        self.hdim = self.model[0].hdim if isinstance(self.model, torch.nn.Sequential) else self.model.hdim
        if mode == 'Rescal':
            self.R = torch.nn.Parameter(torch.randn(len(rel2idx), self.hdim, self.hdim), requires_grad=True)
            self.f = lambda x,r,y: (x * (r @ y.view(y.shape[0], 1, -1).mT).view(y.shape[0], -1)).sum(-1)
            self.prior = { 'head': lambda x,r: (r @ x.view(-1,1,self.hdim)).squeeze(2), 'tail': lambda x,r: (x.view(-1,1,self.hdim) @ r).squeeze(1) }
            self.fast_f = lambda p,y : (p*y).sum(-1)
        else:
            if not external_rel_embs:
                self.R = torch.nn.Parameter(torch.randn(len(rel2idx), self.hdim), requires_grad=True)
            if mode == 'Distmult':
                 self.f = lambda x,r,y : (x*r*y).sum(-1)
                 self.prior = {'head': lambda x,r : x*r, 'tail': lambda x,r : x*r}
                 self.fast_f = lambda p,y: (p*y).sum(-1)
            elif mode == 'TransE':
                 self.f = lambda x,r,y : -((y-x-r)**2).sum(-1) # L2 distance
                 self.prior = {'head': lambda x,r: x-r, 'tail': lambda x,r: x+r}
                 self.fast_f = lambda p,y : -((p-y)**2).sum(-1) # L2 distance # ( this the same for head and tail prediction since (y-x)**2=(x-y)**2)
                 #self.f = lambda x,r,y : -((y-x-r).abs()).sum(-1) # L1 distance
                 #self.fast_f = lambda p,y : -(p-y).abs().sum(-1)
            elif mode == 'ConvE':
                self.f = ConvE(self.hdim, k_w=10, k_h=20)
                self.prior = {'head': None, 'tail': lambda x,r: self.f.prior(x,r)} # how you define the prior in ConvE for head prediction??
                self.fast_f = lambda p,y : self.f.fast_forward(p, y)

    def forward(self, x, y, r): 
        #x_bak, y_bak = x, y
        if self.external_rel_embs:
            xy, rel = self.model(torch.cat((x,y)))
            x, y = xy.view(2,-1, self.hdim)
            del xy
            rel = rel[r]
        else:
            x, y = self.model(torch.cat((x,y))).view(2,-1, self.hdim) # more efficient
            rel = self.R[r]
        #x, y = self.model(x), self.model(y)
        #nanx = x.isnan().any(-1).nonzero()
        #nany = y.isnan().any(-1).nonzero()
        #if len(nanx) > 0:
        #    print('Found nan in x: ')
        #    for i in nanx:
        #        print(x_bak[i])
        #if len(nany) > 0:
        #    print('Found nan in y: ')    # the RGCN is producing nans in some cases
        #    for i in nany:               # in detail entity 8235 seems to be the one causing the nans
        #        print(y_bak[i])
        return self.f(x, rel, y)

    def get_embedding(self, x):
        """Returns the embedding learned by the graph encoder."""
        return self.model(x)

    def score_candidates(self, triples, candidates, mode='tail', filter=None):
        assert mode in ('head','tail')
        if self.mode == 'ConvE':
            assert mode == 'tail' # head prediction not implemented for ConvE
        idx, idx_pair = (2, [0,1]) if mode == 'tail' else (0, [1,2])
        mask = (triples[:,idx].view(-1,1) == candidates) 
        if filter != None:
            filter_mask = (triples.view(-1,1,3)[:,:,idx_pair].detach().cpu() == filter[:,idx_pair]).all(-1)
            tmp_cand = candidates.detach().cpu()
            #idx = 2 if mode == 'tail' else 0
            filter_mask = torch.vstack([
                (filter[filter_mask[i]][:,idx].view(-1,1) == tmp_cand).sum(0).bool()
                for i in range(filter_mask.shape[0])
            ])
            filter_mask = (mask.logical_not() * filter_mask.to(mask.device)).bool()
            del tmp_cand
        else:
            filter_mask = torch.zeros(triples.shape[0], candidates.shape[0]).bool()
        if self.external_rel_embs:
            ht, rel = self.get_embedding(torch.cat((triples[:,0], triples[:,2])))
            h, t = ht.view(2,-1, self.hdim)
            del ht
            r = rel[triples[:,1]]
        else:
            h, t = self.get_embedding(torch.cat((triples[:,0], triples[:,2]))).view(2,-1, self.hdim)
            r = self.R[triples[:,1]]
        prior = self.prior[mode](h, r) if mode == 'tail' else self.prior[mode](t, r)
        prior = torch.hstack([prior for i in range(len(candidates))]).view(-1, prior.shape[-1])
        if self.external_rel_embs:
            candidates, _ = self.get_embedding(candidates)
        else:
            candidates = self.get_embedding(candidates)
        candidates = torch.vstack([candidates for i in range(triples.shape[0])])
        scores = self.fast_f(prior, candidates).view(triples.shape[0], -1)
        if filter != None:
            filter_scores = scores.clone()
            filter_scores[filter_mask] = -1e8 # really small value to move everything at the back
        else:
            filter_scores = None
        return mask, scores, filter_scores
    
class ConcatModel(torch.nn.Module):

    def __init__(self, *models):
        super().__init__()
        self.models = models

    def forward(self, x):
        return torch.cat([ m(x) for m in self.models ], dim=-1)

    @property
    def hdim(self):
        hdim = 0
        for m in self.models:
            hdim += m[0].hdim if isinstance(m, torch.nn.Sequential) else m.hdim
        return hdim

class RGCN(torch.nn.Module):

    def __init__(self, kg, n_layers, indim, hdim, rel_regularizer='basis', num_bases=None, activation = ReLU(), regularization = Dropout(0.2)):
        super().__init__()
        assert rel_regularizer in {'bdd', 'basis'}
        self.kg = kg
        self._hdim = hdim
        self.layers = torch.nn.ModuleList()
        act = [activation for i in range(n_layers-1)] + [None]
        self.layers.append(RelGraphConv(indim, hdim, kg.n_rel, regularizer=rel_regularizer, num_bases=num_bases,
                                    activation=act[0], layer_norm=regularization))
        for a in act[1:]:
            self.layers.append(RelGraphConv(hdim, hdim, kg.n_rel, regularizer=rel_regularizer, num_bases=num_bases,
                                       activation=a, layer_norm=regularization))
        
    def forward(self, nodes):
        h = self.kg.node_feat.weight
        #print(f'Initial Node Features:\n{h}')
        for l in self.layers:
            h = l(self.kg.g, h, self.kg.etypes)
        return h[nodes]

    @property
    def hdim(self):
        return self._hdim

class CompGCN(torch.nn.Module):

    def __init__(self, kg, n_layers, indim, hdim, num_bases=-1, comp_fn='sub', activation = ReLU(), regularization = Dropout(0.2)):
        super().__init__()
        self.kg = kg
        self._indim = indim
        self._hdim = hdim
        self.num_rel = kg.n_rel
        self.num_bases = num_bases
        self.layers = torch.nn.ModuleList()
        act = [activation for i in range(n_layers-1)] + [None]
        self.layers.append(CompGraphConv(indim, hdim, comp_fn=comp_fn))
        for a in act[1:]:
            self.layers.append(CompGraphConv(hdim, hdim, comp_fn=comp_fn))

        # Initial relation embeddings
        if self.num_bases > 0:
            self.basis = torch.nn.Parameter(torch.Tensor(self.num_bases, self._indim), requires_grad=True)
            self.weights = torch.nn.Parameter(torch.Tensor(self.num_rel, self.num_bases), requires_grad=True)
            torch.nn.init.xavier_normal_(self.basis)
            torch.nn.init.xavier_normal_(self.weights)
        else:
            self.rel_embds = torch.nn.Parameter(torch.Tensor(self.num_rel, self._indim), requires_grad=True)
            torch.nn.init.xavier_normal_(self.rel_embds)

        # Dropout after compGCN layers
        self.dropouts = torch.nn.ModuleList()
        for i in range(n_layers):
            self.dropouts.append(regularization)
        
    def forward(self, nodes):
        h = self.kg.node_feat.weight
        r = torch.mm(self.weights, self.basis) if self.num_bases > 0 else self.rel_embds
        #print(f'Initial Node Features:\n{h}')
        # add dropout
        for l,d in zip(self.layers, self.dropouts):
            h, r = l(self.kg.g, h, r)
            h = d(h)
        return h[nodes], r

    @property
    def hdim(self):
        return self._hdim

class CompGCNWrapper(torch.nn.Module):

    def __init__(self, kg, n_layers, indim, hdim, num_bases=-1, comp_fn='sub', return_rel_embs=True):
        super().__init__()
        self._hdim = hdim
        self.kg = kg
        self.model = compgcn.CompGCN(
            num_bases = num_bases,
            num_rel = kg.n_rel,
            num_ent = kg.g.num_nodes(),
            in_dim = indim,
            layer_size = [hdim for i in range(n_layers)],
            comp_fn = comp_fn,
            batchnorm = True,
            dropout = 0.1,
            layer_dropout = [0.3 for i in range(n_layers)]
        )
        self.return_rel_embs = return_rel_embs

    def forward(self, nodes):
        node_feat, rel_feat = self.model(self.kg.g)
        if self.return_rel_embs:
            return node_feat[nodes], rel_feat
        else:
            return node_feat[nodes]
        
    @property
    def hdim(self):
        return self._hdim

class ConvE(torch.nn.Module):

    def __init__(self, hdim, k_w, k_h, hid_drop=0.3, feat_drop=0.3, ker_sz=5, num_filt=200):
        super(ConvE, self).__init__()

        assert k_w*k_h == hdim
        self.embed_dim = hdim
        self.hid_drop = hid_drop
        self.feat_drop = feat_drop
        self.ker_sz = ker_sz
        self.k_w = k_w
        self.k_h = k_h
        self.num_filt = num_filt

        # batchnorms to the combined (sub+rel) emb
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        # dropouts and conv module to the combined (sub+rel) emb
        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(
            1,
            out_channels=self.num_filt,
            kernel_size=(self.ker_sz, self.ker_sz),
            stride=1,
            padding=0,
            bias=False,
        )

        flat_sz_h = int(2 * self.k_w) - self.ker_sz + 1
        flat_sz_w = self.k_h - self.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)

    # combine entity embeddings and relation embeddings
    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.k_w, self.k_h)
        )
        return stack_inp

    def prior(self, x, r):
        # combine the sub_emb and rel_emb
        stk_inp = self.concat(x, r)
        # use convE to score the combined emb
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        return x
    
    def forward(self, h, r, t):
        x = self.prior(h, r)
        return (x*t).sum(-1)

    def fast_forward(self, prior, t):
        return (prior*t).sum(-1)
