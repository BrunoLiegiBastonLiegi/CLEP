import re, torch, time
import numpy as np
from torch.nn import Linear, BatchNorm1d, Dropout, ReLU, Sequential
from torch.nn.functional import normalize
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, BertModel, AutoModel
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

    def __init__(self, graph_encoder, text_encoder, hdim: int, head_to_tail=False):
        super().__init__()
        self.hdim = hdim
        self.h_to_t = head_to_tail
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
        #self.t_nn = Sequential(self.t_encoder, self.t_mlp)
        # graph encoding
        self.g_encoder = graph_encoder
        self.g_mlp = MLP(1, self.g_encoder.hdim, hdim)
        #self.g_mlp = TransformerEncoder(1, self.g_encoder.hdim, hdim)
        #self.g_nn = Sequential(self.g_encoder, self.g_mlp)

    def forward(self, nodes, captions):
        self.T = min(self.T, 100)
        if self.h_to_t:
            ents, rel = self.g_encoder(nodes[:,0])
            rel = rel[nodes[:,1]]
            nodes = ents + rel
            nodes = self.g_mlp(nodes)
        else:
            nodes = self.g_mlp(self.g_encoder(nodes))
        captions = self.t_mlp(self.t_encoder(captions))
        return normalize(nodes, p=2, dim=-1), normalize(captions, p=2, dim=-1)
        #return ( normalize(self.g_nn(nodes), p=2, dim=-1),
        #         normalize(self.t_nn(captions), p=2, dim=-1) )

class PretrainedGraphEncoder(torch.nn.Module):

    def __init__(self, node_embeddings: dict, index: dict, device: torch.device):
        super().__init__()
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
        del embs

    def forward(self, nodes):
        return self.ordered_embs[nodes].squeeze(1).to(self.dev) # dinamically move to device the batch
    
    @property
    def hdim(self):
        #return self.ordered_embs.shape[-1]
        return self._hdim


class CaptionEncoder(torch.nn.Module):

    def __init__(self, pretrained_model: str = 'gpt2'):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)      
        for m in self.model.layers[:-4]: # freezing every layer but the last 4
            for p in m.parameters():
                p.requires_grad = False
        
    def forward(self, x, span=None):
        if span is None:
            return self.model(**x).last_hidden_state[:,-1,:]
        else:
            return self.model(**x).last_hidden_state[:,span[0]:span[1],:]

    @property
    def hdim(self):
        return next(self.model.layers[-1].parameters()).shape[-1]
    
    
class GPT2CaptionEncoder(torch.nn.Module):

    def __init__(self, pretrained_model: str = 'gpt2'):
        super().__init__()
        #self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.model = GPT2Model.from_pretrained(pretrained_model)       # which one is better to use?
        for m in self.model.h[:-4]: # freezing every layer but the last 4
            for p in m.parameters():
                p.requires_grad = False
        
    def forward(self, x, span=None):
        #return self.model(**x).logits[:,-1,:]
        if span is None:
            return self.model(**x).last_hidden_state[:,-1,:]
        else:
            return self.model(**x).last_hidden_state[:,span[0]:span[1],:]

    @property
    def hdim(self):
        #return self.model.config.vocab_size
        return self.model.config.n_embd

class BertCaptionEncoder(torch.nn.Module):

    def __init__(self, pretrained_model: str = 'bert-base-cased'):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        try:
            layers = self.model.encoder.layer[:-4]
        except:
            layers = self.model.transformer.layer[:-4]
        for m in layers: # freezing every layer but the last 4
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

    def __init__(self, graph_embedding_model, rel2idx : dict, mode: str = 'Distmult', external_rel_embs=False, one_to_N_scoring=False):
        super().__init__()
        assert mode in ('Distmult', 'TransE', 'Rescal', 'ConvE')
        self.mode = mode
        self.external_rel_embs = external_rel_embs
        self.one_to_N = one_to_N_scoring
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
                self.f = Distmult(one_to_N_scoring=self.one_to_N)
            elif mode == 'TransE':
                self.f = lambda x,r,y : -((y-x-r)**2).sum(-1) # L2 distance
                self.prior = {'head': lambda x,r: x-r, 'tail': lambda x,r: x+r}
                self.fast_f = lambda p,y : -((p-y)**2).sum(-1) # L2 distance # ( this the same for head and tail prediction since (y-x)**2=(x-y)**2)
            elif mode == 'ConvE':
                self.f = ConvE(self.hdim, k_w=10, k_h=20, one_to_N_scoring=one_to_N_scoring)

    def forward(self, x, y, r):
        if self.external_rel_embs:
            node_embs, rel = self.model(self.model.kg.g.nodes())
            rel = rel[r]
        else:
            node_embs = self.model(self.model.kg.g.nodes())
            rel = self.R[r]
        x, y = node_embs[x], node_embs[y]
        if self.one_to_N:
            t_scores, h_scores = self.f(x, rel, node_embs), self.f(node_embs, rel, y)
            #return t_scores
            return torch.vstack((t_scores, h_scores))
        else:
            return self.f(x, rel, y)

    def get_embedding(self, x):
        """Returns the embedding learned by the graph encoder."""
        return self.model(x)

    def score_candidates(self, triples, candidates, mode='tail', filter=None):
        assert mode in ('head','tail')
        if not self.one_to_N:
            self.f.one_to_N = True
        if self.mode == 'ConvE' and mode == 'head':
            print('# Warning: head prediction with ConvE.')
            #assert mode == 'tail' # head prediction not implemented for ConvE
        idx, idx_pair = (2, [0,1]) if mode == 'tail' else (0, [1,2])
        mask = (triples[:,idx].view(-1,1) == candidates) 
        if filter != None:
            filter_mask = (triples.view(-1,1,3)[:,:,idx_pair] == filter[:,idx_pair]).all(-1)
            filter_mask = torch.vstack([
                (filter[filter_mask[i]][:,idx].view(-1,1) == candidates).sum(0).bool()
                for i in range(filter_mask.shape[0])
            ])
            filter_mask = (mask.logical_not() * filter_mask.to(mask.device)).bool()
        else:
            filter_mask = torch.zeros(triples.shape[0], candidates.shape[0]).bool()
        if self.external_rel_embs:
            node_embs, rel = self.get_embedding(self.model.kg.g.nodes())
            r = rel[triples[:,1]]
        else:
            node_embs = self.get_embedding(self.model.kg.g.nodes())
            r = self.R[triples[:,1]]
        h, t = node_embs[triples[:,0]], node_embs[triples[:,2]]
        #scores = self.f(h, r, node_embs) if mode == 'tail' else self.f(t, r, node_embs)
        scores = self.f(h, r, node_embs) if mode == 'tail' else self.f(node_embs, r, t)
        if not self.one_to_N:
            self.f.one_to_N = False
        if filter != None:
            filter_scores = scores.clone()
            filter_scores[filter_mask] = -1e8 # really small value to move everything at the back
        else:
            filter_scores = None

        return mask, scores, filter_scores

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
            
        self.node_feats = torch.nn.Parameter(torch.Tensor(len(kg.e2idx), indim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.node_feats)
        
    def forward(self, nodes):
        h = self.node_feats
        for l in self.layers:
            h = l(self.kg.g, h, self.kg.etypes)
        return h[nodes]

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

    def __init__(self, hdim, k_w, k_h, one_to_N_scoring=False, hid_drop=0.3, feat_drop=0.3, ker_sz=7, num_filt=200):
        super(ConvE, self).__init__()

        assert k_w*k_h == hdim
        self.embed_dim = hdim
        self.hid_drop = hid_drop
        self.feat_drop = feat_drop
        self.ker_sz = ker_sz
        self.k_w = k_w
        self.k_h = k_h
        self.one_to_N = one_to_N_scoring
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
        if self.one_to_N:
            if h.shape[0] == r.shape[0]:
                x = self.prior(h, r)
                return x.mm(t.T)
            elif t.shape[0] == r.shape[0]:
                x = self.prior(t, r)
                return x.mm(h.T)
        else:
            x = self.prior(h, r)
            return (x*t).sum(-1)

    def fast_forward(self, prior, t):
        return (prior*t).sum(-1)

class Distmult(torch.nn.Module):

    def __init__(self, one_to_N_scoring=False):
        super(Distmult, self).__init__()
        self.one_to_N = one_to_N_scoring

    def forward(self, h, r, t):
        if self.one_to_N:
            if h.shape[0] == r.shape[0]:
                return (h*r).mm(t.T)
            elif t.shape[0] == r.shape[0]:
                return (t*r).mm(h.T)
        else:
            return (h*r*t).sum(-1)


class EntityLinkingModel(torch.nn.Module):

    def __init__(self, clep_model, tokenizer):
        super().__init__()
        self.clep_model = clep_model
        self.tokenizer = tokenizer
        self.dev = None

    def forward(self, entity_mention, entity, top_k=1):
        if self.dev is None:
            self.dev = next(self.clep_model.g_mlp.parameters()).device
        # edit the sentence to help the tokenizer
        # insert white space between contiguos punctuation: ., -> . ,
        entity_mention = re.sub("(?<=[.,:;\)])(?=[.,:;])", r"\g<0> ", entity_mention.lower())
        # insert white space in expression between apices: "xxx" -> " xxx "
        entity_mention = re.sub("(?<=[\s\(][\"])[^\"]+(?=[\"][\s\)])", r" \g<0> ", entity_mention)

        if entity_mention[0] != " ":
            entity_mention = f" {entity_mention}"
        tokenized_mention = self.tokenizer(entity_mention, add_special_tokens=False, return_tensors="pt").to(self.dev)
        tokenized_entity = self.tokenizer(entity.lower(), add_special_tokens=False, return_tensors="pt").to(self.dev)
        span = self.find_entity_span(tokenized_mention.input_ids, tokenized_entity.input_ids)
        if span is None:
            raise RuntimeError(f"Entity `{entity}` not found in sentence `{entity_mention}`.")
        text_embedding = self.clep_model.t_encoder(tokenized_mention, span).mean(1).reshape(1, 1, -1)
        # this tests the use of the last token of the mention as identifier of the entity, but it seems to work worse 
        #text_embedding = self.clep_model.t_encoder(tokenized_mention, None).reshape(1, 1, -1)
        text_embedding = self.clep_model.t_mlp(text_embedding)
        graph_embedding = self.clep_model.g_encoder(None)
        graph_embedding = self.clep_model.g_mlp(graph_embedding)
        scores, node_indices = torch.nn.functional.cosine_similarity(text_embedding.squeeze(0), graph_embedding.squeeze(0)).sort(descending=True)
        return node_indices[:top_k]

    def find_entity_span(self, entity_mention, entity, allow_recursion=True):
        l = entity.shape[-1]
        i = 0
        while i + l <= entity_mention.shape[-1]:
            if all(entity_mention[0][i:i+l] == entity[0]):
                return (i, i+l)
            i += 1
        ent = self.tokenizer.decode(entity.ravel())
        # try with a space in front
        if ent[0] != " " and allow_recursion:
            ent = self.tokenizer(f" {ent.lower()}", add_special_tokens=False, return_tensors="pt").to(self.dev).input_ids
            return self.find_entity_span(entity_mention, ent)
        # some labels don't precisely coincide with the words in the text
        else:
            # they miss the final s, n or ed for instance
            desinences = ("s", "n", f"{ent[-1]}ed", "ic", "en", "es", "ns", "er", "ation", "ing", "ed", f"{ent[-1]}ing", "al", "\"", "ern", "h", "e", "te", "ian", "tic", "an", "rs", "nese", "lary", "vian", "ans")
            for desinence in desinences:
                if ent[-1] != desinence and allow_recursion:
                    span = self.find_entity_span(
                        entity_mention,
                        self.tokenizer(f"{ent}{desinence}", add_special_tokens=False, return_tensors="pt").to(self.dev).input_ids,
                        False
                    )
                    if span is not None:
                        return span
    
        
