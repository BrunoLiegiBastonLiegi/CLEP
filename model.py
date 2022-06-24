import torch
import numpy as np
from torch.nn import Linear, BatchNorm1d, Dropout, ReLU, Sequential
from torch.nn.functional import normalize
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

class MLP(torch.nn.Module):

    def __init__(self, n_layers: int, indim: int, hdim: int, activation = ReLU(), normalization = Dropout(0.2)):
        super().__init__()
        self.n_layers = n_layers
        self.indim = indim
        self.hdim = hdim
        layers = [Linear(indim, hdim)]
        for n in range(n_layers-1):
            layers.append(normalization)
            layers.append(activation)
            layers.append(Linear(hdim, hdim))
        self.nn = Sequential(*layers)

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
            dim_feedforward = hdim,
            batch_first = True
        )
        self.nn = torch.nn.TransformerEncoder(self.layer, n_layers)

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
        self.t_mlp = MLP(1, self.t_encoder.hdim, hdim) # Switch dropout with BatchNorm!!
        # graph encoding
        self.g_encoder = graph_encoder
        self.g_mlp = MLP(2, self.g_encoder.hdim, hdim)
        #self.g_mlp = TransformerEncoder(2, self.g_encoder.hdim, hdim)

    def forward(self, nodes, captions):
        self.T = min(self.T, 100)
        return ( normalize(self.g_mlp(self.g_encoder(nodes)), p=2, dim=-1),
                 normalize(self.t_mlp(self.t_encoder(captions)), p=2, dim=-1) )

class PretrainedGraphEncoder(torch.nn.Module):

    def __init__(self, node_embeddings: dict, device: torch.device):
        super().__init__()
        self.node2emb = node_embeddings
        self.dev = device
        self.register_parameter(
            name='ordered_embs',
            param=torch.nn.Parameter(
                torch.vstack(list(zip(
                    *sorted(node_embeddings.items(),
                            key=lambda x: x[0])
                ))[1]),
                requires_grad = False
            )
        )
        #self.ordered_embs = torch.vstack(list(zip(
        #    *sorted(node_embeddings.items(),
        #            key=lambda x: x[0])
        #))[1])

    def forward(self, nodes):
        #return self.ordered_embs[nodes].squeeze(1).float().to(self.dev) # need to explictly cast to float32
        return self.ordered_embs[nodes].squeeze(1).float()
    
    @property
    def hdim(self):
        #return next(iter(self.node2emb.items()))[1].shape[-1]
        return self.ordered_embs.shape[-1]
        
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

    #def unfreeze_layer(self, i: int):
     #   for p in self.model.

        
