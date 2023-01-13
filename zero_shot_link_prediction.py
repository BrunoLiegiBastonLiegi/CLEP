import argparse, torch, json, pickle, time, random
from transformers import GPT2Tokenizer, BertTokenizer
from torch.utils.data import DataLoader
from dataset import LinkPredictionDataset
from model import LinkPredictionModel, PretrainedGraphEncoder, MLP, CLIP_KB, GPT2CaptionEncoder, BertCaptionEncoder, RGCN, CompGCNWrapper
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils import  KG
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--dataset', default=None)
parser.add_argument('--train_data', help='Path to train data file.')
parser.add_argument('--test_data', help='Path to test data file.')
parser.add_argument('--valid_data', help='Path to test data file.')
parser.add_argument('--entity_index', default=None, help='Path to entity index file.')
parser.add_argument('--rel_index', help='Path to relations index file.')
parser.add_argument('--entities', help='Path to captions index file.')
parser.add_argument('--load_model', help='Path to caption pretrained model.')
parser.add_argument('--graph', default=None, help='Path to graph triples file.')
parser.add_argument('--save_results', default='lp_results.json')
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
    args.entities = 'data/{}/entities.json'.format(args.dataset)

print('---------------- Arguments -----------------')
for k,v in vars(args).items():
    print(f'{k}: {v}')
print('--------------------------------------------')
    
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

filter_triples = torch.cat([train_data.triples, test_data.triples, valid_data.triples])[:,:3].to(dev)

if args.graph != None:
    kg = KG(ent2idx=wid2idx, rel2idx=rel2idx, embedding_dim = 200, dev=dev, add_inverse_edges=add_inverse)
    kg.build_from_file(args.graph)
else:
    triples = train_data.true_triples if train_data.inv_triples == None else torch.vstack((train_data.true_triples, train_data.inv_triples))
    kg = KG(triples = triples, ent2idx=wid2idx, rel2idx=rel2idx, embedding_dim = 200, dev=dev)
nodes = kg.g.nodes()

for d in (train_data, test_data, valid_data):
    d.triples = d.true_triples

print('> Initializing Caption Encoder.')
t_encoder = GPT2CaptionEncoder(pretrained_model='gpt2')

print('> Initializing Graph Encoder.')
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
g_encoder = CompGCNWrapper(**conf)
baseline = CompGCNWrapper(**conf) # Just use a randomly initialized CompGCN as baseline

print('> Loading Pretrained CLIP Model.')
clip = CLIP_KB(
    graph_encoder = g_encoder,
    text_encoder = t_encoder,
    hdim = 200
).to(dev)
clip.load_state_dict(torch.load(args.load_model))
try:
    clip.g_encoder.return_rel_embs = True
except:
    print('# Warning: the graph encoder does not returns relation embeddings.')

class CaptionEncodingData(Dataset):

    def __init__(self, captions, ids, tokenizer):
        self.ids = ids
        self.captions = list(zip(ids,captions))
        self.tok = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i):
        return self.captions[i]

    def collate_fn(self, batch):
        ids, captions = [], []
        for item in batch:
            ids.append(item[0])
            captions.append(item[1])
        captions = self.tok(text=captions, padding=True, return_tensors='pt')
        return captions, torch.as_tensor(ids)

    def get_loader(self, batchsize=128):
        return DataLoader(self.captions, batch_size=batchsize, shuffle=False, collate_fn=self.collate_fn)

with open(args.entities, 'r') as f:
    id2cap = {}
    for v in json.load(f).values():
        k = wid2idx[v['entity_id']]
        if v['caption'] is None:
            id2cap.update({k: 'Caption not available.'})
        else:
            id2cap.update({k: v['caption']})
    ids, cap = list(zip(*id2cap.items()))


print('> Loading Tokenizer.')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side, tokenizer.pad_token = 'left', tokenizer.bos_token

data = CaptionEncodingData(captions=cap, ids=ids, tokenizer=tokenizer)

print('> Encoding Entity Captions.')
index, caption_encodings = [], []
bs = 64
for batch in tqdm(data.get_loader(batchsize=bs)):
    with autocast() and torch.no_grad():
        captions, ids = batch[0].to(dev), batch[1].to(dev)
        captions = clip.t_nn(captions)
        caption_encodings.append(captions)
        index.append(ids)

index, caption_encodings = torch.cat(index), torch.nn.functional.normalize(torch.cat(caption_encodings), p=2, dim=-1)
# REMEMBER TO NORMALIZE CAPTIONS AND NODE ENCODINGS BEFORE COMPARING/MAKING OPERATIONS ON THEM <------------
        
LP_loader = DataLoader(
    test_data,
    batch_size = 32,
    shuffle = True,
    collate_fn = test_data.collate_fn
)

print('> Zero-shot Link Prediction.')
ranks = []
for batch, _ in tqdm(LP_loader):
    with torch.no_grad() and autocast():
        triples = batch.to(dev)
        tail_mask = (triples[:,2].view(-1,1) == index) 
        mask = (triples.view(-1,1,3)[:,:,[0,1]] == filter_triples[:,[0,1]]).all(-1)
        mask = torch.vstack([
            (filter_triples[mask[i]][:,2].view(-1,1) == index).sum(0).bool()
            for i in range(mask.shape[0])
        ])
        mask = (tail_mask.logical_not() * mask.to(tail_mask.device)).bool()
        h, r, t = triples[:,0], triples[:,1], triples[:,2]
        h, rel = clip.g_encoder(h)
        r = rel[r]
        h = torch.nn.functional.normalize(clip.g_mlp(h + r), p=2, dim=-1)
        # Normalization ?? Is it needed?
        #scores = ((h.view(batch.shape[0],1,-1) - caption_encodings)**2).sum(-1).sqrt()
        scores = (h.view(batch.shape[0],1,-1) * caption_encodings).sum(-1)
        scores[mask] = 1e8
        #prediction = index[scores.sort(-1)[1]]
        prediction = index[scores.sort(-1, descending=True)[1]]
        ranks.append((t.view(-1,1) == prediction).nonzero()[:,1])
        
ranks = torch.cat(ranks).view(-1) + 1

metrics = {
    'mrr': (1/ranks).mean(dtype=float).item(),
    'mean_rank': ranks.mean(dtype=float).item(),
    'hits@1': len((ranks == 1).nonzero()) / len(ranks),
    'hits@3': len((ranks <= 3).nonzero()) / len(ranks),
    'hits@10': len((ranks <= 10).nonzero()) / len(ranks)
}

LPmodel = LinkPredictionModel(
        graph_embedding_model = baseline,
        mode = 'Distmult',
        #mode = 'TransE',
        #mode = 'Rescal',
        #mode = 'ConvE',
        rel2idx = rel2idx,
        external_rel_embs = True,
        one_to_N_scoring = True
    ).to(dev)

def eval_f(model, data):
    global filter_triples
    global nodes

    #data.triples = data.true_triples
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
    return metrics, ranks['right']['filtered']

baseline_metrics, baseline_ranks = eval_f(LPmodel, test_data)

print('--- CompGCN Baseline ---')
print(json.dumps({k:v for k, v in baseline_metrics['filtered'].items() if 'right' in k}, indent=2))
print('--- CP Pretrained CompGCN ---')
print(json.dumps(metrics, indent=2))

bins = 100
dens = True
plt.hist(baseline_ranks.cpu().numpy(), bins=bins, density=dens, alpha=0.5)
plt.hist(ranks.cpu().numpy(), bins=bins, density=dens, alpha=0.5)
#plt.yscale('log')
plt.xlabel('rank')
plt.savefig('zero_shot_lp_{}.pdf'.format(args.dataset), dpi=300, format='pdf')
plt.show()
