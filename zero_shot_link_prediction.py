import argparse, torch, json, pickle, time, random
from dataset import LinkPredictionDataset
from model import LinkPredictionModel, PretrainedGraphEncoder, MLP, CLIP_KB, GPT2CaptionEncoder, BertCaptionEncoder, RGCN, CompGCNWrapper
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from utils import  KG
from torch.utils.data import Dataset


parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--train_data', help='Path to train data file.')
parser.add_argument('--test_data', help='Path to test data file.')
parser.add_argument('--valid_data', help='Path to test data file.')
parser.add_argument('--entity_index', default=None, help='Path to entity index file.')
parser.add_argument('--rel_index', help='Path to relations index file.')
parser.add_argument('--id2cap', help='Path to captions index file.')
parser.add_argument('--load_model', help='Path to caption pretrained model.')
parser.add_argument('--graph', default=None, help='Path to graph triples file.')
parser.add_argument('--save_results', default='lp_results.json')
parser.add_argument('--one_to_N_scoring', action='store_true')
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
valid_data = LinkPredictionDataset(
    datafile = args.valid_data,
    entity2idx = wid2idx,
    rel2idx = rel2idx,
    add_inverse_edges = add_inverse
)

rel2idx = train_data.r2idx

w = 2 # number of corrupted triples per positive triple
filter_triples = torch.cat([train_data.triples, test_data.triples, valid_data.triples])[:,:3].to(dev)

if not args.one_to_N_scoring:
    if add_inverse:
        train_data.generate_corrupted_triples('data/FB15k-237/link-prediction/corrupted_train_triples.pt', mode='load')
        test_data.generate_corrupted_triples('data/FB15k-237/link-prediction/corrupted_test_triples.pt', mode='load')
        valid_data.generate_corrupted_triples('data/FB15k-237/link-prediction/corrupted_valid_triples.pt', mode='load')
        #train_data.generate_corrupted_triples('data/WN18RR/link-prediction/corrupted_train_triples+inverse.pt', mode='load')
        #test_data.generate_corrupted_triples('data/WN18RR/link-prediction/corrupted_test_triples+inverse.pt', mode='load')
        #valid_data.generate_corrupted_triples('data/WN18RR/link-prediction/corrupted_valid_triples+inverse.pt', mode='load')
    else:
        train_data.generate_corrupted_triples('data/FB15k-237/corrupted_train+valid_triples.pt', mode='load')
        test_data.generate_corrupted_triples('data/FB15k-237/corrupted_test_triples.pt', mode='load')

if args.graph != None:
    kg = KG(ent2idx=wid2idx, rel2idx=rel2idx, embedding_dim = 200, dev=dev, add_inverse_edges=add_inverse)
    kg.build_from_file(args.graph)
else:
    triples = train_data.true_triples if train_data.inv_triples == None else torch.vstack((train_data.true_triples, train_data.inv_triples))
    kg = KG(triples = triples, ent2idx=wid2idx, rel2idx=rel2idx, embedding_dim = 200, dev=dev)
nodes = kg.g.nodes()

t_encoder = GPT2CaptionEncoder(pretrained_model='gpt2')

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

clip = CLIP_KB(
    graph_encoder = g_encoder,
    text_encoder = t_encoder,
    hdim = 200
).to(dev)
clip.load_state_dict(torch.load(args.load_model))
try:
    clip.graph_encoder.return_rel_embs = True
except:
    print('Warnging: the graph encoder does not returns relation embeddings.')

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
        return Dataloader(self.captions, batch_size=batchsize, shuffle=False, collate_fn=self.collate_fn)

with open(args.id2cap, 'r') as f:
    ids, cap = list(zip(*json.load(id2cap).item()))
    ids = [ wid2idx[i] for i in ids ]
    
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side, tokenizer.pad_token = 'left', tokenizer.bos_token

data = CaptionEncodingData(captions=cap, ids=ids, tokenizer=tokenizer)

caption_encodings = {}
for batch in tqdm(data.get_loader()):
    with torch.autocast() and torch.no_grad():
        captions, ids = batch[0].to(dev), batch[1].to(dev)
        captions = clip.t_nn(captions)
        caption_encodings.update(dict(zip(ids, captions)))
print(caption_encodings)

index, caption_encodings = caption_encodings.items()
index, caption_encodings = torch.as_tensor(index), torch.vstack(caption_encodings)
# REMEMBER TO NORMALIZE CAPTIONS AND NODE ENCODINGS BEFORE COMPARING/MAKING OPERATIONS ON THEM <------------
        
LP_loader = Dataloader(
    test_data,
    batch_size = 512,
    shuffle = True,
    collate_fn = test_data.collate_fn
)

ranks = []
for batch, _ in tqdm(LP_loader()):
    with torch.no_grad() and torch.autocast():
        triples = batch.to(dev)
        h, r, t = triples[:,0], triples[:,1], triples[:,2]
        h, r = clip.g_encoder(h, r)
        h = clip.g_mlp(h + r)
        # Normalization ?? Is it needed?
        distances = ((h.view(batch.shape[0],1,-1) - caption_encodings)**2).sum(-1).sqrt()
        prediction = index[distances.sort(-1)[1]]
        ranks.append((t.view(-1,1) == prediction).nonzero()[:,1])
        
ranks = torch.cat(ranks).view(-1) + 1

metrics = {
    'mrr': (1/ranks).mean(dtype=float).item(),
    'mean_rank': ranks.mean(dtype=float).item(),
    'hits@1': len((ranks == 1).nonzero()) / len(ranks),
    'hits@3': len((ranks <= 3).nonzero()) / len(ranks),
    'hits@10': len((ranks <= 10).nonzero()) / len(ranks)
}

print(metrics)

