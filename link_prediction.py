import argparse, torch, json, pickle, time, random
from dataset import LinkPredictionDataset
from model import LinkPredictionModel, PretrainedGraphEncoder, MLP, CLIP_KB, GPT2CaptionEncoder, BertCaptionEncoder, ConcatModel, RGCN, CompGCN, CompGCNWrapper
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics import F1Score
from utils import training_routine, KG
from multiprocessing import Pool
from itertools import repeat
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--train_data', help='Path to train data file.')
parser.add_argument('--test_data', help='Path to test data file.')
parser.add_argument('--graph_embeddings', help='Path to pretrained embeddings file.')
parser.add_argument('--entity_index', default=None, help='Path to entity index file.')
parser.add_argument('--rel_index', help='Path to relations index file.')
parser.add_argument('--load_model', help='Path to caption pretrained model.')
parser.add_argument('--graph', default=None, help='Path to graph triples file.')
parser.add_argument('--save_results', default='lp_results.json')
args = parser.parse_args()

# Set device for computation
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
else:
    dev = torch.device('cpu')
print(f'\n> Setting device {dev} for computation.')

# Load index
with open(args.entity_index, 'r') as f:
    wid2idx = json.load(f)
with open (args.rel_index, 'r') as f:
    rel2idx = json.load(f)

add_inverse = True

# Train and Test data
train_data = LinkPredictionDataset(
    datafile = args.train_data, 
    entity2idx = wid2idx,
    rel2idx = rel2idx,
    add_inverse_edges = add_inverse
)
test_data = LinkPredictionDataset(
    datafile = args.test_data,
    entity2idx = wid2idx,
    rel2idx = rel2idx,
    add_inverse_edges = add_inverse
)
val_data = LinkPredictionDataset(
    datafile = 'data/FB15k-237/link-prediction/_valid_wiki-id.txt',
    entity2idx = wid2idx,
    rel2idx = rel2idx,
    add_inverse_edges = add_inverse
)

rel2idx = train_data.r2idx

w = 2 # number of corrupted triples per positive triple
#filter_triples = torch.cat([train_data.true_triples, test_data.true_triples, val_data.true_triples])[:,:3]
#filter_triples = torch.cat([train_data.true_triples, test_data.true_triples])[:,:3]
filter_triples = torch.cat([train_data.triples, test_data.triples])[:,:3]
#train_data.generate_corrupted_triples(filter_triples, mode='gen', w=int(w/2))
#test_data.generate_corrupted_triples(filter_triples, mode='gen', w=int(w/2))
if add_inverse:
    train_data.generate_corrupted_triples('data/FB15k-237/corrupted_train+valid_triples+inverse.pt', mode='load')
    test_data.generate_corrupted_triples('data/FB15k-237/corrupted_test_triples+inverse.pt', mode='load')
else:
    train_data.generate_corrupted_triples('data/FB15k-237/corrupted_train+valid_triples.pt', mode='load')
    test_data.generate_corrupted_triples('data/FB15k-237/corrupted_test_triples.pt', mode='load')


if args.graph_embeddings != None:
    with open(args.graph_embeddings, 'rb') as f:
        node_embeddings = pickle.load(f)
# Baseline: pretrained TransE embeddings
#BaselineModel = PretrainedGraphEncoder(node_embeddings=node_embeddings, index=wid2idx, device=dev)

if  args.graph != None:
    triples = train_data.true_triples if train_data.inv_triples == None else torch.vstack((train_data.true_triples, train_data.inv_triples))
    #triples = train_data.true_triples
    kg = KG(triples = triples, rel2idx=rel2idx, embedding_dim = 200, dev=dev)
    #kg.build_from_file(args.graph, wid2idx, rel2idx)
    kg.node_feat = torch.load('data/FB15k-237/initial_node_features.pt')
nodes = kg.g.nodes()

rgcn_conf = {
    'kg': kg,
    'n_layers': 2,
    'indim': kg.embedding_dim,
    'hdim': 200,
    'rel_regularizer': 'basis',
    #'rel_regularizer': 'bdd',
    'num_bases': 64
}

BaselineModel = RGCN(**rgcn_conf)
"""
BaselineModel = CompGCN(
    kg,
    2,
    kg.embedding_dim,
    200,
    num_bases = 5,
    comp_fn = 'mul'
)
"""
"""
BaselineModel = CompGCNWrapper(
    kg = kg,
    n_layers = 1,
    #n_layers = 3,
    indim = kg.embedding_dim,
    hdim = 200,
    num_bases = -1,
    #comp_fn = 'sub',
    comp_fn = 'ccorr',
    return_rel_embs = True
)
"""
# Caption prediction pretraining
# Annoyingly I have to load the gpt model to load the weights I need, even though I am not
# going to use that. A possible solution would be to save the complete model instead of saving
# just the state_dict, that would require more disk space though.
# REMEMBER: I NEED TO RERUN PRETRAINING SINCE wid2idx.json HAS CHANGED

_ = GPT2CaptionEncoder(pretrained_model='gpt2')
#_ = BertCaptionEncoder(pretrained_model='bert-base-cased')
"""
clip = CLIP_KB(
    graph_encoder = RGCN(**rgcn_conf),
    text_encoder = _,
    hdim = 200
).to(dev)
clip.load_state_dict(torch.load(args.load_model))
# Stack the pretrained MLP on top of the graph embedding model
#SemanticAugmentedModel = torch.nn.Sequential(clip.g_encoder, clip.g_mlp)
SemanticAugmentedModel = clip.g_nn
"""

#for par in SemanticAugmentedModel.parameters():
#    par.requires_grad = False
#ConcatModel = ConcatModel(BaselineModel, SemanticAugmentedModel)


# Training
# Define training step
def step_f(model, batch, label, dev):
    batch, label = batch.to(dev), label.to(dev)
    out = model(batch[:,0], batch[:,2], r=batch[:,1])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out,
        label,
        weight = torch.ones(batch.shape[0], device=dev)*1/(1+w) # w=2 ratio negative/positive triples (c = 1/(w+1))
    )
    del out, batch, label
    torch.cuda.empty_cache()
    return loss

# Define evaluation function
def eval_f(model, data):
    global filter_triples
    global nodes

    data.triples = data.true_triples
    dataloader = DataLoader(
        data,
        batch_size = 128,
        shuffle = True,
        collate_fn = test_data.collate_fn
    )
    
    ranks = {'raw': [], 'filtered': []}
    for i, (batch, _) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            mask, raw_scores, filter_scores = model.score_candidates(
                triples = batch.to(dev),
                candidates = nodes,
                mode = 'tail',
                filter = filter_triples
            )
            ranks['raw'].append((raw_scores.sort(dim=-1, descending=True, stable=False).indices == mask.nonzero()[:,1].view(-1,1)).nonzero()[:,1])
            ranks['filtered'].append((filter_scores.sort(dim=-1, descending=True, stable=False).indices == mask.nonzero()[:,1].view(-1,1)).nonzero()[:,1])

    ranks['raw'] = torch.cat(ranks['raw']).view(-1) + 1 # +1 since the position starts counting from zero
    ranks['filtered'] = torch.cat(ranks['filtered']).view(-1) + 1 # +1 since the position starts counting from zero
    return {
        k: {
            'mrr': (1/v).mean(dtype=float).item(),
            'mean_rank': v.mean(dtype=float).item(),
            'hits@1': len((v == 1).nonzero()) / len(v),
            'hits@3': len((v <= 3).nonzero()) / len(v),
            'hits@10': len((v <= 10).nonzero()) / len(v)
        }
        for k,v in ranks.items()
    }

def experiment(model, train_data, test_data, dev=dev, rel2idx=rel2idx):
    # build LP model
    LPmodel = LinkPredictionModel(
        graph_embedding_model = model,
        mode = 'Distmult',
        #mode = 'TransE',
        #mode = 'Rescal',
        #mode = 'ConvE',
        rel2idx = rel2idx,
        external_rel_embs = True
        #external_rel_embs = False
    ).to(dev)
    # train
    epochs = 5
    #batchsize = 128
    batchsize = 128
    #batchsize = 256
    #batchsize = 8192
    #batchsize = 32768
    lr = 1e-3 
    train_loss, test_loss, metrics = training_routine(
        model = LPmodel,
        step_f = step_f,
        train_data = train_data,
        test_data = test_data,
        epochs = epochs,
        batchsize = batchsize,
        learning_rate = lr,
        eval_f = eval_f,
        eval_each = 1,# epochs, # evaluate the metrics each n epoch/s
        accum_iter = 1,
        dev = dev
    )
    return metrics


# Latent space visualilzation
from utils import visualize_embeddings
import matplotlib.pyplot as plt
test_loader = DataLoader(
        test_data,
        batch_size = 8192,
        shuffle = False,
        collate_fn = test_data.collate_fn
    )

def get_embeddings(model, loader):
    embs = {}
    for i, (batch, _) in enumerate(loader):
        print(f'{i}/{len(test_loader)}', end='\r')
        with torch.no_grad():
            batch = batch.view(-1,1).to(dev)
            out = model(batch)
            embs.update(dict(zip(batch.flatten().detach().cpu().tolist(),out.detach().cpu())))
    return embs

fig, ax = plt.subplots(1,2, figsize=(24,16))

#embs = get_embeddings(SemanticAugmentedModel, test_loader)
#clusters = visualize_embeddings(torch.vstack(list(embs.values())), n_clusters=50, ax=ax[0])

# Finetuning
results = {'RGCN with Caption Pretraining': {}, 'RGCN Baseline': {}}
#for m, name in zip((SemanticAugmentedModel, BaselineModel), results.keys()):
for m in (BaselineModel,):
    results[m] = experiment(
        model = m,
        train_data = train_data,
        test_data = test_data,
        dev = dev,
        rel2idx = rel2idx
    )

#embs = get_embeddings(SemanticAugmentedModel, test_loader)
#visualize_embeddings(torch.vstack(list(embs.values())), ax=ax[1])
#plt.show()

with open(args.save_results, 'w') as f:
    json.dump(results, f, indent=2)

