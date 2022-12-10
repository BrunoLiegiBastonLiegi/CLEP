import torch, argparse, json, random, time, pickle, numpy
from dataset import CLIPDataset, LinkPredictionDataset
from model import CLIP_KB, PretrainedGraphEncoder, GPT2CaptionEncoder, BertCaptionEncoder, RGCN, CompGCNWrapper
from transformers import GPT2Tokenizer, BertTokenizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from tqdm import tqdm
from utils import training_routine, KG
from torch.utils.data import Dataset


parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--dataset', default=None)
parser.add_argument('--train_data', default=None, help='Path to train data file.')
parser.add_argument('--test_data', default=None, help='Path to test data file.')
parser.add_argument('--graph_embeddings', default=None, help='Path to the pretrained graph embedding file.')
parser.add_argument('--entity_index', default=None, help='Path to relations index file.')
parser.add_argument('--rel_index', default=None, help='Path to relations index file.')
parser.add_argument('--load_model', default=None, help='Path to caption pretrained model.')
parser.add_argument('--graph', default=None, help='Path to graph triples file.')
parser.add_argument('--head_to_tail', action='store_true')
parser.add_argument('--entities', help='Path to entities file.')
parser.add_argument('--batchsize', help='Batchsize.', default=128, type=int)
parser.add_argument('--save_model', help='Save model to.')
parser.add_argument('--graph_encoder', default='RGCN')
parser.add_argument('--epochs', help='Epochs.', default=32, type=int)

args = parser.parse_args()

if args.dataset is not None:
    args.entity_index = 'data/{}/ent2idx.json'.format(args.dataset)
    args.rel_index = 'data/{}/rel2idx.json'.format(args.dataset)
    args.entities = 'data/{}/pretraining/entities.json'.format(args.dataset)
    args.graph = 'data/{}/link-prediction/train.txt'.format(args.dataset)
    if args.head_to_tail:
        args.train_data = 'data/{}/link-prediction/train.txt'.format(args.dataset)
        args.test_data = 'data/{}/link-prediction/test.txt'.format(args.dataset)
        args.val_data = 'data/{}/link-prediction/valid.txt'.format(args.dataset)
    else:
        args.train_data = 'data/{}/pretraining/train.json'.format(args.dataset)
        args.test_data = 'data/{}/pretraining/test.json'.format(args.dataset)

if args.save_model is None:
    args.save_model = '{}_{}bs_{}e_{}'.format(args.graph_encoder, args.batchsize, args.epochs, args.dataset)
    if args.head_to_tail:
        args.save_model += '_h_to_t'
    args.save_model += '.pt'

print(args.save_model)
        
# Set device for computation
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
else:
    dev = torch.device('cpu')
print(f'\n> Setting device {dev} for computation.')

# Choose the tokenizer
print(f'> Loading Pretrained tokenizer.')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side, tokenizer.pad_token = 'left', tokenizer.bos_token
#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

print('> Preparing the data.')
# Load index mapping
with open(args.entity_index, 'r') as f:
    wid2idx = json.load(f)
with open (args.rel_index, 'r') as f:
    rel2idx = json.load(f)
        
# Train and Test data
if args.head_to_tail:
    
    train_triples = LinkPredictionDataset(
        datafile = args.train_data, 
        entity2idx = wid2idx,
        rel2idx = rel2idx,
        add_inverse_edges = True
    )
    train_data = CLIPDataset(
        datafile = args.entities,
        tokenizer = tokenizer,
        entity2idx = wid2idx,
        triples = train_triples.triples,
        device = dev
    )

    test_triples = LinkPredictionDataset(
        datafile = args.test_data, 
        entity2idx = wid2idx,
        rel2idx = rel2idx,
        add_inverse_edges = True
    )
    test_data = CLIPDataset(
        datafile = args.entities,
        tokenizer = tokenizer,
        entity2idx = wid2idx,
        triples = test_triples.triples,
        device = dev
    )

    valid_triples = LinkPredictionDataset(
        datafile = args.val_data, 
        entity2idx = wid2idx,
        rel2idx = rel2idx,
        add_inverse_edges = True
    )

    filter_triples = torch.cat([train_triples.triples, test_triples.triples, valid_triples.triples])[:,:3].to(dev)

else:
    train_data = CLIPDataset(
        datafile = args.train_data,
        tokenizer = tokenizer,
        entity2idx = wid2idx,
        device = dev
    )
    test_data = CLIPDataset(
        datafile = args.test_data,
        tokenizer = tokenizer,
        entity2idx = wid2idx,
        device = dev
    )

print('> Initializing the model.')
# Graph encoder
if  args.graph_embeddings != None:
    with open(args.graph_embeddings, 'rb') as f:
        node_embeddings = pickle.load(f)
        
inverse_edges = True
if  args.graph != None:
    kg = KG(embedding_dim=200, ent2idx=wid2idx, rel2idx=rel2idx, dev=dev, add_inverse_edges=inverse_edges)
    kg.build_from_file(args.graph)
else:
    try:
        kg = KG(triples = train_triples, ent2idx=wid2idx, rel2idx=rel2idx, embedding_dim = 200, dev=dev)
    except:
        assert False, 'No data provided for building the graph, try using the --graph argument.'

#graph_encoder = PretrainedGraphEncoder(node_embeddings=node_embeddings, index=wid2idx, device=dev)

graph_model = args.graph_encoder
if graph_model == 'CompGCN':
    conf = {
        'kg': kg,
        'n_layers': 2,
        'indim': kg.embedding_dim,
        'hdim': 200,
        'num_bases': -1,
        'comp_fn' : 'sub',
        'return_rel_embs':  args.head_to_tail
    }
    graph_encoder = CompGCNWrapper(**conf)
    
elif graph_model == 'RGCN':
    conf = {
        'kg': kg,
        'n_layers': 2,
        'indim': kg.embedding_dim,
        'hdim': 200,
        'rel_regularizer': 'basis',
        #'rel_regularizer': 'bdd',
        'num_bases': 64
    }
    graph_encoder = RGCN(**conf)

if args.head_to_tail:
    assert graph_model == 'CompGCN' and graph_encoder.return_rel_embs, "Head-to-Tail pretraining is only supported for CompGCN models with return_rel_embs=True"
# Caption encoder
text_encoder = GPT2CaptionEncoder(pretrained_model='gpt2')
#text_encoder = BertCaptionEncoder(pretrained_model='bert-base-cased')
# CLIP
model = CLIP_KB(
    graph_encoder=graph_encoder,
    text_encoder=text_encoder,
    hdim=200,
    head_to_tail=args.head_to_tail
).to(dev)

#original_node_feat = graph_encoder.model.n_embds.clone().cpu()

# Training

# Define training step
def step_f(model, batch, label, dev):
    label = label.to(dev)
    graph_out, text_out = model(batch['entities'].to(dev), batch['captions'].to(dev))
    logits = torch.tensordot(graph_out, text_out.T, dims=1) * torch.exp(model.T)
    del graph_out
    del text_out
    torch.cuda.empty_cache()
    loss = 0.5 * ( torch.nn.functional.cross_entropy(logits, label) + torch.nn.functional.cross_entropy(logits.T, label) )
    del logits
    torch.cuda.empty_cache()
    return loss

# Define Evaluation
def eval_f(model, data):
    global test_triples, filter_triples, dev
    
    LP_loader = DataLoader(
        test_triples,
        batch_size = 32,
        shuffle = True,
        collate_fn = data.collate_fn
    )
    
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

    capdata = CaptionEncodingData(list(data.idx2cap.values()), ids=list(data.idx2cap.keys()), tokenizer=data.tok)
    index, caption_encodings = [], []
    with torch.no_grad() and autocast():
        for batch in tqdm(capdata.get_loader(batchsize=64)):
            captions, ids = batch[0].to(dev), batch[1].to(dev)
            captions = model.t_nn(captions)
            caption_encodings.append(captions)
            index.append(ids)
    index, caption_encodings = torch.cat(index), torch.nn.functional.normalize(torch.cat(caption_encodings), p=2, dim=-1)
            
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
            h, rel = model.g_encoder(h)
            r = rel[r]
            h = torch.nn.functional.normalize(model.g_mlp(h + r), p=2, dim=-1)
            # Normalization ?? Is it needed?
            distances = ((h.view(batch.shape[0],1,-1) - caption_encodings)**2).sum(-1).sqrt()
            distances[mask] = 1e8
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
    return metrics
    

if args.load_model == None:
    epochs = args.epochs
    batchsize = args.batchsize
    
    training_routine(
        model = model,
        step_f = step_f,
        eval_f = eval_f if args.head_to_tail else None,
        eval_each = 1,
        train_data = train_data,
        test_data = test_data,
        epochs = epochs,
        batchsize = batchsize,
        learning_rate = 5e-4,
        accum_iter = 1,
        dev = dev
    )
    model_name = '{}_{}bs_{}e_{}.pt'.format(args.graph_encoder, args.batchsize, args.epochs, args.dataset) if args.save_model is None else args.save_model#input('> Save model to:\n\t')
    torch.save(model.state_dict(), model_name)
    #torch.save(model.state_dict(),
    #           '{}_fb15k237_{}_layers-{}_{}-{}_epochs.pt'.format(
    #               model_name, rgcn_conf['n_layers'], rgcn_conf['rel_regularizer'], rgcn_conf['num_bases'], epochs)
    #           )

    #print(graph_encoder.model.n_embds.cpu() - original_node_feat)
else:
    model.load_state_dict(torch.load(args.load_model))

batchsize = 256

test_loader = DataLoader(
        test_data,
        batch_size = batchsize,
        shuffle = True,
        collate_fn = test_data.collate_fn
    )

with torch.no_grad():
    sm = torch.nn.Softmax(1)
    acc, tot = 0, 0
    on_diag_dist, off_diag_dist = [], []
    original_points, points, entities, captions = [], [], [], []
    fig, ax = plt.subplots(1,1)
    model.eval()
    for batch, label in test_loader:
        graph_out, text_out = model(batch['entities'].to(dev), batch['captions'].to(dev))
        original_points.append(graph_encoder(batch['entities'])[0].detach().cpu())
        points.append(graph_out.detach().cpu()) # points for space visualization
        entities.append(batch['entities'].detach().cpu())
        captions += tokenizer.batch_decode(batch['captions']['input_ids'].detach().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # Distance of correct pairs
        on_diag_dist.append(((graph_out-text_out)**2).sum(-1).sqrt())
        # Distance of offdiagonal pairs
        idx = set(range(graph_out.shape[0]))
        j = random.sample(list(idx), k=graph_out.shape[0]) # randomly sample a subset of offdiagonal pairs
        k = list(map(lambda x: random.choice(list(idx-{x})), j))
        off_diag_dist.append(((graph_out[j]-text_out[k])**2).sum(-1).sqrt())
        #off_diag_dist.append(((graph_out[k]-text_out[j])**2).sum(-1).sqrt()) # asymettric in principle
        # Accuracy
        logits = torch.tensordot(graph_out, text_out.T, dims=1) #* torch.exp(model.T)
        for i, v in enumerate(sm(logits)):
            tot += 1
            if torch.argmax(v) == i:
                acc += 1
    on_diag_dist = torch.cat(on_diag_dist).detach().cpu().numpy()
    off_diag_dist = torch.cat(off_diag_dist).detach().cpu().numpy()
    #print(ttest_ind(distance, off_diag_dist, equal_var=False))
    #print(mannwhitneyu(distance, off_diag_dist))
    #print(mannwhitneyu(distance, off_diag_dist, alternative='less'))
    #print(mannwhitneyu(distance, off_diag_dist, alternative='greater'))
    #print(sorted(distance))
    print(f'left mean: {on_diag_dist.mean()}\t right mean: {off_diag_dist.mean()}')
    print(f'Distance of the means: {off_diag_dist.mean() - on_diag_dist.mean():.3f}')
    # Get area of histogram overlap
    hist_range = (0.,2.)
    bins = 100
    on_diag_hist, _, _ = ax.hist(on_diag_dist, bins=bins, range=hist_range, alpha=0.5, density=True)
    off_diag_hist, _, _ = ax.hist(off_diag_dist, bins=bins, range=hist_range, alpha=0.5, density=True)
    area = []
    for on, off in zip(on_diag_hist, off_diag_hist):
        if on > 0 and off > 0:
            area.append(min(on, off))
    area = (torch.as_tensor(area) * (hist_range[1] - hist_range[0])/100).sum()
    print(f'Overlapping area: {area:.3f}')
    ax.annotate(f'Left mean: {on_diag_dist.mean():.2f}     Right mean: {off_diag_dist.mean():.2f}', (0.1,0.9), xycoords='axes fraction')
    ax.annotate(f'Overlapping area: {area:.3f}', (0.1,0.8), xycoords='axes fraction')
    #plt.savefig(f'distance_histogram_batchsize_{batchsize}.png')
    plt.show()
    print(f'> {acc} correct out of {tot} ({acc/tot*100:.2f}%).')

    # Latent space visualization
    from utils import visualize_embeddings
    fig, ax = plt.subplots(1,2, figsize=(24,16))
    points = torch.vstack(points).numpy()
    original_points = torch.vstack(original_points).numpy()
    entities = torch.vstack(entities).numpy().flatten()
    n_clusters = 30 # 50-60 appears to be the optimal number
    c1 = visualize_embeddings(points, n_clusters, ax[0])
    c2 = visualize_embeddings(original_points, n_clusters, ax[1])
    plt.show()
    
    clusters_1 = {i:[] for i in range(n_clusters)}
    clusters_2 = {i:[] for i in range(n_clusters)}
    idx2wid = dict(zip(wid2idx.values(), wid2idx.keys()))
    for e, c, cap in zip(entities, c1, captions):
        clusters_1[c].append((idx2wid[e], cap))
    for e, c, cap in zip(entities, c2, captions):
        clusters_2[c].append((idx2wid[e], cap))
    for k,v in clusters_1.items():
        print(f'# CLUSTER {k}')
        for c in random.choices(v, k=5):
            print(c)

