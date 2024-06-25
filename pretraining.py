from sklearn.utils import validation
import torch, argparse, json, random, time, pickle, numpy, os
from dataset import CLIPDataset, LinkPredictionDataset
from model import CLIP_KB, PretrainedGraphEncoder, GPT2CaptionEncoder, CaptionEncoder, RGCN, CompGCNWrapper
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer, DistilBertTokenizerFast
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from scipy.stats import ttest_ind, mannwhitneyu
from tqdm import tqdm
from utils import training_routine, KG
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP


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
parser.add_argument('--text_encoder', default='gpt2')
parser.add_argument('--use_valid_data', action='store_true')
parser.add_argument('--initial_node_embeddings', help='path to intial embeddings')
parser.add_argument('--add_label_to_caption', action='store_true')




args = parser.parse_args()

if args.dataset is not None:
    args.entity_index = 'data/{}/ent2idx.json'.format(args.dataset)
    args.rel_index = 'data/{}/rel2idx.json'.format(args.dataset)
    args.entities = 'data/{}/entities.json'.format(args.dataset)
    args.graph = 'data/{}/link-prediction/train.txt'.format(args.dataset)
    args.initial_node_embeddings = 'data/{}/pretrained_entity_embeddings.json'.format(args.dataset)
    if args.head_to_tail:
        args.train_data = 'data/{}/link-prediction/train.txt'.format(args.dataset)
        args.test_data = 'data/{}/link-prediction/test.txt'.format(args.dataset)
        args.val_data = 'data/{}/link-prediction/valid.txt'.format(args.dataset)
    else:
        args.train_data = 'data/{}/pretraining/train.json'.format(args.dataset)
        args.test_data = 'data/{}/pretraining/test.json'.format(args.dataset)
        args.dev_data = 'data/{}/pretraining/dev.json'.format(args.dataset)

if args.save_model is None:
    args.save_model = 'saved/models/{}/pretraining/{}/{}-{}_{}bs_{}e_{}'.format(
        args.dataset,
        args.graph_encoder,
        args.graph_encoder,
        args.text_encoder.replace("/", "-"),
        args.batchsize,
        args.epochs,
        args.dataset
    )
    if args.head_to_tail:
        args.save_model += '_h_to_t'
    args.save_model += '.pt'

print(f'> Saving model to: {args.save_model}')
        
# Set device for computation
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
else:
    dev = torch.device('cpu')
print(f'\n> Setting device {dev} for computation.')


# Choose the tokenizer
print(f'> Loading Pretrained tokenizer.')
tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
#tokenizer = GPT2Tokenizer.from_pretrained(args.text_encoder)
if "gpt" in args.text_encoder:
    tokenizer.padding_side = 'left'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
#tokenizer = DistilBertTokenizerFast.from_pretrained(args.text_encoder)

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
    test_triples = LinkPredictionDataset(
        datafile = args.test_data, 
        entity2idx = wid2idx,
        rel2idx = rel2idx,
        add_inverse_edges = True
    )
    valid_triples = LinkPredictionDataset(
        datafile = args.val_data, 
        entity2idx = wid2idx,
        rel2idx = rel2idx,
        add_inverse_edges = True
    )
    filter_triples = torch.cat([train_triples.triples, test_triples.triples, valid_triples.triples])[:,:3].to(dev)
    
    train_data = CLIPDataset(
        datafile = args.entities,
        tokenizer = tokenizer,
        entity2idx = wid2idx,
        triples = train_triples.triples,
        filter_triples = train_triples.triples[:,:3].to(dev),
        device = dev
    )
    test_data = CLIPDataset(
        datafile = args.entities,
        tokenizer = tokenizer,
        entity2idx = wid2idx,
        triples = test_triples.triples,
        filter_triples = filter_triples,
        device = dev
    )

else:
    train_data = CLIPDataset(
        datafile = args.train_data,
        tokenizer = tokenizer,
        entity2idx = wid2idx,
        device = dev,
        concatenate_labels = args.add_label_to_caption,
    )
    test_data = CLIPDataset(
        datafile = args.test_data,
        tokenizer = tokenizer,
        entity2idx = wid2idx,
        device = dev,
        concatenate_labels = args.add_label_to_caption,
    )
    try:
        valid_data = CLIPDataset(
            datafile = args.test_data,
            tokenizer = tokenizer,
            entity2idx = wid2idx,
            device = dev,
            concatenate_labels = args.add_label_to_caption,
        )
    except:
        print("> No valid data found, skipping it.")
    if args.use_valid_data:
        train_data.data += valid_data.data
    

print('> Initializing the model.')
# Graph encoder
if  args.graph_embeddings != None:
    with open(args.graph_embeddings, 'rb') as f:
        node_embeddings = pickle.load(f)
        
inverse_edges = True
if args.graph != None:
    kg = KG(embedding_dim=200, ent2idx=wid2idx, rel2idx=rel2idx, dev=dev, add_inverse_edges=inverse_edges)
    kg.build_from_file(args.graph)
else:
    try:
        kg = KG(triples = train_triples, ent2idx=wid2idx, rel2idx=rel2idx, embedding_dim = 200, dev=dev)
    except:
        assert False, 'No data provided for building the graph, try using the --graph argument.'

#graph_encoder = PretrainedGraphEncoder(node_embeddings=node_embeddings, index=wid2idx, device=dev)

if args.initial_node_embeddings is not None:
    try:
        with open(args.initial_node_embeddings, 'r') as f:
            initial_node_embeddings = json.load(f)
        initial_node_embeddings = [initial_node_embeddings[e] for e,i in sorted(wid2idx.items(), key=lambda x: x[1])]
    except FileNotFoundError:
        initial_node_embeddings = None

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
        'num_bases': 64,
        'initial_embeddings': initial_node_embeddings,
    }
    graph_encoder = RGCN(**conf)

if args.head_to_tail:
    assert graph_model == 'CompGCN' and graph_encoder.return_rel_embs, "Head-to-Tail pretraining is only supported for CompGCN models with return_rel_embs=True"
# Caption encoder
if "gpt2" in args.text_encoder:
    text_encoder = GPT2CaptionEncoder(pretrained_model=args.text_encoder)
else:
    text_encoder = CaptionEncoder(pretrained_model=args.text_encoder)

# CLIP
model = CLIP_KB(
    graph_encoder=graph_encoder,
    text_encoder=text_encoder,
    hdim=200,
    head_to_tail=args.head_to_tail
).to(dev)
#model = DDP(model, device_ids=rank)

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

if args.head_to_tail:

    LP_loader = DataLoader(
        valid_triples,
        batch_size = 256,
        shuffle = True,
        collate_fn = valid_triples.collate_fn
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

    capdata = CaptionEncodingData(list(test_data.idx2cap.values()), ids=list(test_data.idx2cap.keys()), tokenizer=test_data.tok)
    
# Define Evaluation
def eval_f(model, data):
    global test_triples, filter_triples, dev, capdata
    
    index, caption_encodings = [], []
    for batch in tqdm(capdata.get_loader(batchsize=64)):
        with torch.no_grad() and autocast():
            captions = model.t_mlp(model.t_encoder(batch[0].to(dev))).detach()
            caption_encodings.append(captions)
            del captions
            torch.cuda.empty_cache()
            index.append(batch[1].to(dev))
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
            #scores = ((h.view(batch.shape[0],1,-1) - caption_encodings)**2).sum(-1).sqrt() # L1 distance 
            scores = (h.view(batch.shape[0],1,-1) * caption_encodings).sum(-1) # cosine similarity
            #scores[mask] = 1e8
            scores[mask] = -1
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
    return metrics

def unfreezing_f(model, epoch):
    if epoch > 1:
        model.t_encoder.unfreeze_layers(4)

if args.load_model == None:
    epochs = args.epochs
    batchsize = args.batchsize
    
    _, _, metrics = training_routine(
        model = model,
        step_f = step_f,
        eval_f = eval_f if args.head_to_tail else None,
        unfreezing_f=unfreezing_f,
        eval_each = 1,
        train_data = train_data,
        test_data = test_data,
        epochs = epochs,
        batchsize = batchsize,
        learning_rate = 5e-4,
        accum_iter = 1,
        dev = dev
    )
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    torch.save(model.state_dict(), args.save_model)
    if args.head_to_tail:
        LP_loader = DataLoader(
            test_triples,
            batch_size = 256,
            shuffle = True,
            collate_fn = valid_triples.collate_fn
        )
        metrics['test'] = eval_f(model, None)
        res_file = 'saved/LP_results/{}/CompGCN/head_to_tail/lp_results_CompGCN_{}bs_{}e_{}_h_to_t.json'.format(args.dataset, args.batchsize, args.epochs, args.dataset)
        print(f'> Saving LP results to {res_file}.')
        os.makedirs(os.path.dirname(res_file), exist_ok=True)
        with open(res_file, 'w') as f:
            json.dump(metrics, f, indent=2)
else:
    model.load_state_dict(torch.load(args.load_model))

batchsize = args.batchsize#256

test_loader = DataLoader(
        test_data,
        batch_size = 256,#batchsize,
        shuffle = True,
        collate_fn = test_data.collate_fn
    )

with torch.no_grad():
    sm = torch.nn.Softmax(1)
    acc, tot = 0, 0
    on_diag_dist, off_diag_dist = [], []
    fig, ax = plt.subplots(1,1)
    model.eval()
    for batch, label in test_loader:
        graph_out, text_out = model(batch['entities'].to(dev), batch['captions'].to(dev))
        # Distance of correct pairs
        on_diag_dist.append(((graph_out-text_out)**2).sum(-1).sqrt())
        #on_diag_dist.append((graph_out * text_out).sum(-1).sqrt())
        # Distance of offdiagonal pairs
        idx = set(range(graph_out.shape[0]))
        j = random.sample(list(idx), k=graph_out.shape[0]) # randomly sample a subset of offdiagonal pairs
        #print(idx)
        #print(j)
        k = list(map(lambda x: random.choice(list(idx-{x})), j))
        off_diag_dist.append(((graph_out[j]-text_out[k])**2).sum(-1).sqrt())
        #off_diag_dist.append((graph_out[j] * text_out[k]).sum(-1).sqrt())
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
    #import matplotlib.patches as mpatches
    #on_diag_patch = mpatches.Patch(color='lightgreen', label=r'$P\bigg(\|\tilde{x}_i^{(g)}-\tilde{x}_i^{(t)}\|\bigg)$')
    #off_diag_patch = mpatches.Patch(color='salmon', label=r'$P\bigg(\|\tilde{x}_i^{(g)}-\tilde{x}_j^{(t)}\|_{i\neq j}\bigg)$')
    #plt.legend(handles=[on_diag_patch, off_diag_patch])
    print(f'left mean: {on_diag_dist.mean()}\t right mean: {off_diag_dist.mean()}')
    print(f'Distance of the means: {off_diag_dist.mean() - on_diag_dist.mean():.3f}')
    # Get area of histogram overlap
    hist_range = (0.,2.)
    bins = 100
    on_diag_hist, _, _ = ax.hist(
        on_diag_dist,
        bins=bins,
        range=hist_range,
        alpha=0.5,
        density=True,
        color='mediumseagreen',
        label=r'$P(\|\tilde{x}_i^{(g)}-\tilde{x}_i^{(t)}\|)$'
    )
    off_diag_hist, _, _ = ax.hist(
        off_diag_dist,
        bins=bins,
        range=hist_range,
        alpha=0.5,
        density=True,
        color='salmon',
        label=r'$P(\|\tilde{x}_i^{(g)}-\tilde{x}_j^{(t)}\|_{i\neq j})$'
    )
    ax.legend()
    area = []
    for on, off in zip(on_diag_hist, off_diag_hist):
        if on > 0 and off > 0:
            area.append(min(on, off))
    area = (torch.as_tensor(area) * (hist_range[1] - hist_range[0])/100).sum()
    print(f'Overlapping area: {area:.3f}')
    plt.axvline(on_diag_dist.mean(), linestyle='--', alpha=0.5, c='mediumseagreen')
    plt.axvline(off_diag_dist.mean(), linestyle='--', alpha=0.5, c='salmon')
    ax.set_xlabel(r'$\|\tilde x^{(g)} - \tilde x^{(t)}\|$')
    #ax.annotate(f'Left mean: {on_diag_dist.mean():.2f}     Right mean: {off_diag_dist.mean():.2f}', (0.1,0.9), xycoords='axes fraction')
    #ax.annotate(f'Overlapping area: {area:.3f}', (0.1,0.8), xycoords='axes fraction')
    #plt.savefig(f'distance_histogram_batchsize_{batchsize}.png')
    plt.savefig('euclidean_dist_{}.pdf'.format(args.dataset), dpi=300, format='pdf', bbox_inches='tight')
    plt.show()
    print(f'> {acc} correct out of {tot} ({acc/tot*100:.2f}%).')


