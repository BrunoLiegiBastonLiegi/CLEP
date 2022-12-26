import argparse, torch, json, pickle, time, random, numpy, os
from dataset import LinkPredictionDataset
from model import LinkPredictionModel, PretrainedGraphEncoder, MLP, CLIP_KB, GPT2CaptionEncoder, BertCaptionEncoder, RGCN, CompGCNWrapper
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics import F1Score
from utils import training_routine, KG
import matplotlib.pyplot as plt
from os.path import basename

parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--dataset', default=None)
parser.add_argument('--train_data', help='Path to train data file.')
parser.add_argument('--test_data', help='Path to test data file.')
parser.add_argument('--valid_data', help='Path to test data file.')
parser.add_argument('--graph_embeddings', help='Path to pretrained embeddings file.')
parser.add_argument('--entity_index', default=None, help='Path to entity index file.')
parser.add_argument('--rel_index', help='Path to relations index file.')
parser.add_argument('--graph_encoder', default='RGCN')
parser.add_argument('--load_model', help='Path to caption pretrained model.')
parser.add_argument('--graph', default=None, help='Path to graph triples file.')
parser.add_argument('--save_results', default=None)
parser.add_argument('--one_to_N_scoring', action='store_true')
parser.add_argument('--train_corrupted_triples', help='Path to train corrupted triples file.')
parser.add_argument('--test_corrupted_triples', help='Path to test corrupted triples file.')
parser.add_argument('--valid_corrupted_triples', help='Path to valid corrupted triples file.')
parser.add_argument('--LP_head', default='Distmult')
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--seed', type=int)

args = parser.parse_args()

if args.dataset is not None:
    args.entity_index = 'data/{}/ent2idx.json'.format(args.dataset)
    args.rel_index = 'data/{}/rel2idx.json'.format(args.dataset)
    #args.graph = 'data/{}/link-prediction/train.txt'.format(args.dataset)
    args.train_data = 'data/{}/link-prediction/train.txt'.format(args.dataset)
    args.test_data = 'data/{}/link-prediction/test.txt'.format(args.dataset)
    args.valid_data = 'data/{}/link-prediction/valid.txt'.format(args.dataset)
    args.train_corrupted_triples = 'data/{}/link-prediction/corrupted_train_triples+inverse.pt'.format(args.dataset)
    args.test_corrupted_triples = 'data/{}/link-prediction/corrupted_test_triples+inverse.pt'.format(args.dataset)
    args.valid_corrupted_triples = 'data/{}/link-prediction/corrupted_valid_triples+inverse.pt'.format(args.dataset)
    

if args.save_results is None:
    args.save_results = 'saved/LP_results/{}/{}/lp_results_{}_{}+{}_{}bs_{}e'.format(
        args.dataset,
        args.graph_encoder,
        args.dataset,
        args.graph_encoder,
        args.LP_head,
        args.batchsize,
        args.epochs
    )
    if args.one_to_N_scoring:
        args.save_results += '_1_to_N'
    args.save_results += '.json'

print(args)
    
print(f'Saving results to: {args.save_results}')    

if args.seed is not None:
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
print(f'> Seed: {torch.seed()}')

    
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
valid_data = LinkPredictionDataset(
    datafile = args.valid_data,
    entity2idx = wid2idx,
    rel2idx = rel2idx,
    add_inverse_edges = add_inverse
)

rel2idx = train_data.r2idx

w = 2 # number of corrupted triples per positive triple
filter_triples = torch.cat([train_data.triples, test_data.triples, valid_data.triples])[:,:3]
#train_data.generate_corrupted_triples(filter_triples, mode='gen', w=int(w/2))
#test_data.generate_corrupted_triples(filter_triples, mode='gen', w=int(w/2))
#valid_data.generate_corrupted_triples(filter_triples, mode='gen', w=int(w/2))
if not args.one_to_N_scoring:
    for d,a in zip(
            (train_data, test_data, valid_data),
            (args.train_corrupted_triples, args.test_corrupted_triples, args.valid_corrupted_triples)
    ):
        try:
            d.generate_corrupted_triples(a, mode='load')
        except:
            print('> File not found. Generating new corrupted triples file.')
            d.generate_corrupted_triples(filter_triples, mode='gen', w=int(w/2), save=a)

filter_triples = filter_triples.to(dev)
            
if args.graph_embeddings != None:
    with open(args.graph_embeddings, 'rb') as f:
        node_embeddings = pickle.load(f)
# Baseline: pretrained TransE embeddings
#BaselineModel = PretrainedGraphEncoder(node_embeddings=node_embeddings, index=wid2idx, device=dev)

if args.graph != None:
    kg = KG(ent2idx=wid2idx, rel2idx=rel2idx, embedding_dim = 200, dev=dev, add_inverse_edges=add_inverse)
    kg.build_from_file(args.graph)
else:
    triples = train_data.true_triples if train_data.inv_triples == None else torch.vstack((train_data.true_triples, train_data.inv_triples))
    kg = KG(triples = triples, ent2idx=wid2idx, rel2idx=rel2idx, embedding_dim = 200, dev=dev)
nodes = kg.g.nodes()

graph_model = args.graph_encoder
assert graph_model in ('RGCN', 'CompGCN'), 'Unsupported graph encoder, use RGCN or CompGCN instead.'
print(f'> Using {graph_model} as graph encoder.')

if graph_model == 'RGCN':
    conf = {
        'kg': kg,
        'n_layers': 2,
        'indim': kg.embedding_dim,
        'hdim': 200,
        'rel_regularizer': 'basis',
        #'rel_regularizer': 'bdd',
        'num_bases': 64
    }   
    BaselineModel = RGCN(**conf)
elif graph_model == 'CompGCN':
    conf = {
        'kg': kg,
        'n_layers': 2,
        'indim': kg.embedding_dim,
        'hdim': 200,
        'comp_fn': 'sub',
        'num_bases': -1,
        'return_rel_embs' : True
        #'return_rel_embs' : False
    }   
    BaselineModel = CompGCNWrapper(**conf)

# Caption prediction pretraining
# Annoyingly I have to load the gpt model to load the weights I need, even though I am not
# going to use that. A possible solution would be to save the complete model instead of saving
# just the state_dict, that would require more disk space though.

if args.load_model is not None:
    
    _ = GPT2CaptionEncoder(pretrained_model='gpt2')
    #_ = BertCaptionEncoder(pretrained_model='bert-base-cased')
    
    g_encoder = CompGCNWrapper(**conf) if graph_model == 'CompGCN' else RGCN(**conf)
    clip = CLIP_KB(
        graph_encoder = g_encoder,
        text_encoder = _,
        hdim = 200
    ).to(dev)
    clip.load_state_dict(torch.load(args.load_model))
    # Stack the pretrained MLP on top of the graph embedding model
    class vstack(torch.nn.Module):
        def __init__(self, g_encoder, nn):
            super().__init__()
            self.g_encoder = g_encoder
            self.nn = nn
            self.hdim = g_encoder.hdim
            self.kg = g_encoder.kg
        def forward(self, x):
            if graph_model == 'CompGCN' and conf['return_rel_embs']:
                x, rel = self.g_encoder(x)
                return self.nn(x), rel
            else:
                x = self.g_encoder(x)
                return self.nn(x)
    SemanticAugmentedModel = vstack(clip.g_encoder, clip.g_mlp)
    SemanticAugmentedModel.return_rel_embs = True if graph_model == 'CompGCN' and conf['return_rel_embs'] else False

else:
    SemanticAugmentedModel = None

#for par in SemanticAugmentedModel.parameters():
#    par.requires_grad = False
#ConcatModel = ConcatModel(BaselineModel, SemanticAugmentedModel)


# Training
# Define training step
def step_f(model, batch, label, dev):
    if model.one_to_N:
        global nodes, filter_triples
        batch = batch.to(dev)
        label = torch.vstack((batch[:,2].view(-1,1) == nodes, batch[:,0].view(-1,1) == nodes))
        tail_mask = (batch.view(-1,1,3)[:,:,[0,1]] == filter_triples[:,[0,1]]).all(-1)
        tail_mask = torch.vstack([
                (filter_triples[tail_mask[i]][:,2].view(-1,1) == nodes).sum(0).bool()
                for i in range(tail_mask.shape[0])
            ])
        head_mask = (batch.view(-1,1,3)[:,:,[1,2]] == filter_triples[:,[1,2]]).all(-1)
        head_mask = torch.vstack([
                (filter_triples[head_mask[i]][:,0].view(-1,1) == nodes).sum(0).bool()
                for i in range(head_mask.shape[0])
            ])
        mask = torch.vstack((tail_mask, head_mask))
        del tail_mask, head_mask
        torch.cuda.empty_cache()
        mask = (label.logical_not() * mask).logical_not().view(-1)
        out = model(x=batch[:,0], y=batch[:,2], r=batch[:,1]).view(-1)
        out = out[mask]
        label = label.view(-1)[mask]
        del mask
        torch.cuda.empty_cache()
    else:
        batch, label = batch.to(dev), label.to(dev).view(-1)
        out = model(x=batch[:,0], y=batch[:,2], r=batch[:,1]).view(-1)
    weight = None if model.one_to_N else torch.ones(batch.shape[0], device=dev)*1/(1+w) # w=2 ratio negative/positive triples (c = 1/(w+1))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out,
        label.float(),
        weight = weight,
    )
    #loss = torch.nn.functional.cross_entropy(
    #    torch.sigmoid(out),
    #    label.nonzero()[:,1]
    #)
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
        batch_size = 512,
        shuffle = True,
        collate_fn = test_data.collate_fn
    )
    
    ranks = {'left': {'raw': [], 'filtered': []}, 'right': {'raw': [], 'filtered': []}}
    for i, (batch, _) in enumerate(tqdm(dataloader)):
        triples = batch.to(dev)
        with torch.no_grad():
            for mode, side in zip(('head', 'tail'), ('left', 'right')):
                mask, raw_scores, filter_scores = model.score_candidates(
                    triples = triples,
                    candidates = nodes,
                    mode = mode,
                    filter = filter_triples
                )
                raw_scores, filter_scores = torch.sigmoid(raw_scores), torch.sigmoid(filter_scores) 
                ranks[side]['raw'].append((raw_scores.sort(dim=-1, descending=True, stable=False).indices == mask.nonzero()[:,1].view(-1,1)).nonzero()[:,1])
                ranks[side]['filtered'].append((filter_scores.sort(dim=-1, descending=True, stable=False).indices == mask.nonzero()[:,1].view(-1,1)).nonzero()[:,1])

    for side in ('left', 'right'):
        ranks[side]['raw'] = torch.cat(ranks[side]['raw']).view(-1) + 1 # +1 since the position starts counting from zero
        ranks[side]['filtered'] = torch.cat(ranks[side]['filtered']).view(-1) + 1 # +1 since the position starts counting from zero
    left_metrics = { k: {
        'mrr': (1/v).mean(dtype=float).item(),
        'mean_rank': v.mean(dtype=float).item(),
        'hits@1': len((v == 1).nonzero()) / len(v),
        'hits@3': len((v <= 3).nonzero()) / len(v),
        'hits@10': len((v <= 10).nonzero()) / len(v)
    } for k,v in ranks['left'].items()}
    right_metrics = { k: {
        'mrr': (1/v).mean(dtype=float).item(),
        'mean_rank': v.mean(dtype=float).item(),
        'hits@1': len((v == 1).nonzero()) / len(v),
        'hits@3': len((v <= 3).nonzero()) / len(v),
        'hits@10': len((v <= 10).nonzero()) / len(v)
    } for k,v in ranks['right'].items()
                     }
    metrics = { type: {
        k: 0.5 * (right_metrics[type][k] + left_metrics[type][k])
        for k in right_metrics[type].keys()
    } for type in ('raw', 'filtered')
               }
    for d, side in zip((right_metrics, left_metrics), ('right', 'left')):
        for type in ('raw', 'filtered'):
            for k, v in d[type].items():
                metrics[type][side+'_'+k] = v
    return metrics

def experiment(model, train_data, test_data, valid_data, dev=dev, rel2idx=rel2idx):
    # build LP model
    LPmodel = LinkPredictionModel(
        graph_embedding_model = model,
        mode = args.LP_head,
        rel2idx = rel2idx,
        external_rel_embs = conf['return_rel_embs'] if graph_model == 'CompGCN' else False,
        one_to_N_scoring = args.one_to_N_scoring
    ).to(dev)
    # train
    epochs = args.epochs
    batchsize = args.batchsize

    lr = 1e-3
    train_loss, test_loss, metrics = training_routine(
        model = LPmodel,
        step_f = step_f,
        train_data = train_data,
        test_data = test_data,
        epochs = epochs,
        batchsize = batchsize,
        learning_rate = lr,
        valid_data = valid_data,
        eval_f = eval_f,
        eval_each = 1,# epochs, # evaluate the metrics each n epoch/s
        accum_iter = 1,
        dev = dev
    )
    metrics['test'] = eval_f(LPmodel, test_data)
    print('\n###### Test Metrics ######')
    print(json.dumps(metrics['test'], indent=2))

    name = 'saved/models/{}/link-prediction/{}/LP_{}+{}_{}bs_{}e_{}'.format(
        args.dataset,
        args.graph_encoder,
        args.graph_encoder,
        args.LP_head,
        args.batchsize,
        args.epochs,
        args.dataset
    )
    if isinstance(LPmodel.model, vstack):
        name += '_Finetuned_from_{}'.format(basename(args.load_model))
    else:
        name += '_Baseline.pt'
    os.makedirs(os.path.dirname(name), exist_ok=True)
    torch.save(LPmodel.state_dict(), name)
    print(f'> Model saved to: {name}')
    return metrics


# Finetuning
results = {'{} Caption Pretraining'.format(graph_model): {}, '{} Baseline'.format(graph_model): {}}
for m, name in zip((SemanticAugmentedModel, BaselineModel), results.keys()):
    if m is not None:
        results[name] = experiment(
            model = m,
            train_data = train_data,
            test_data = test_data,
            valid_data = valid_data,
            dev = dev,
            rel2idx = rel2idx
        )
    else:
        results[name] = None


os.makedirs(os.path.dirname(args.save_results), exist_ok=True)        
with open(args.save_results, 'w') as f:
    json.dump(results, f, indent=2)

