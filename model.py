import torch
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

class CLIP_KB(torch.nn.Module):

    def __init__(self, graph_encoder, text_encoder, hdim: int):
        super().__init__()
        self.hdim = hdim
        # text encoding
        self.t_encoder = text_encoder
        self.t_mlp = MLP(3, self.t_encoder.hdim, hdim) # Switch dropout with BatchNorm!!
        # graph encoding
        self.g_encoder = graph_encoder
        self.g_mlp = MLP(3, self.g_encoder.hdim, hdim)

    def forward(self, nodes, captions):
        return ( normalize(self.g_mlp(self.g_encoder(nodes)), p=2, dim=-1),
                 normalize(self.t_mlp(self.t_encoder(captions)), p=2, dim=-1) )

class PretrainedGraphEncoder(torch.nn.Module):

    def __init__(self, node_embeddings: dict):
        super().__init__()
        self.node2emb = node_embeddings
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
        #)[1])

    def forward(self, nodes):
        return self.ordered_embs[nodes].squeeze(1).float() # need to explictly cast to float32

    @property
    def hdim(self):
        #return next(iter(self.node2emb.items()))[1].shape[-1]
        return self.ordered_embs.shape[-1]
        
class GPT2CaptionEncoder(torch.nn.Module):

    def __init__(self, pretrained_model: str = 'gpt2'):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        #self.model = GPT2Model.from_pretrained(pretrained_model)       # which one is better to use? 

    def forward(self, x):
        return self.model(**x).logits[:,-1,:]

    @property
    def hdim(self):
        return self.model.config.vocab_size
        #return self.model.config.n_embd     

        
