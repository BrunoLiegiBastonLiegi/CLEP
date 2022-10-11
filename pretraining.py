import torch, argparse, json, random, time, pickle, numpy
from dataset import CLIPDataset
from model import CLIP_KB, PretrainedGraphEncoder, GPT2CaptionEncoder, BertCaptionEncoder, RGCN, CompGCNWrapper
from transformers import GPT2Tokenizer, BertTokenizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from tqdm import tqdm
from utils import training_routine, KG

parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--train_data', help='Path to train data file.')
parser.add_argument('--test_data', help='Path to test data file.')
parser.add_argument('--graph_embeddings', default=None, help='Path to the pretrained graph embedding file.')
parser.add_argument('--entity_index', default=None, help='Path to relations index file.')
parser.add_argument('--rel_index', default=None, help='Path to relations index file.')
parser.add_argument('--load_model', default=None, help='Path to caption pretrained model.')
parser.add_argument('--graph', default=None, help='Path to graph triples file.')

args = parser.parse_args()

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
if args.rel_index != None:
    with open (args.rel_index, 'r') as f:
        rel2idx = json.load(f)
# Train and Test data
train_data = CLIPDataset(
    datafile = args.train_data, #'data/FB15k-237/train_new.pkl',
    tokenizer = tokenizer,
    entity2idx = wid2idx,
    device = dev
)
test_data = CLIPDataset(
    datafile = args.test_data, #'data/FB15k-237/test_new.pkl',
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
    kg = KG(embedding_dim=200, dev=dev, add_inverse_edges=inverse_edges)
    kg.build_from_file(args.graph, wid2idx, rel2idx)
    #torch.save(kg.node_feat, 'rgcn_initial_node_features.pt')
    kg.node_feat = torch.load('data/FB15k-237/rgcn_initial_node_features.pt')
    
#graph_encoder = PretrainedGraphEncoder(node_embeddings=node_embeddings, index=wid2idx, device=dev)

rgcn_conf = {
    'kg': kg,
    'n_layers': 2,
    'indim': kg.embedding_dim,
    'hdim': 200,
    'rel_regularizer': 'basis',
    #'rel_regularizer': 'bdd',
    'num_bases': 64
}
graph_encoder = RGCN(**rgcn_conf)
graph_encoder = CompGCNWrapper(
    kg = kg,
    n_layers = 2,
    indim = kg.embedding_dim,
    hdim = 200,
    num_bases = 5,
    comp_fn = 'sub',
    return_rel_embs = False
)

# Caption encoder
text_encoder = GPT2CaptionEncoder(pretrained_model='gpt2')
#text_encoder = BertCaptionEncoder(pretrained_model='bert-base-cased')
# CLIP
model = CLIP_KB(graph_encoder=graph_encoder, text_encoder=text_encoder, hdim=200).to(dev)

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

if args.load_model == None:
    epochs = 10
    batchsize = 200
    #batchsize = 128
    
    training_routine(
        model = model,
        step_f = step_f,
        train_data = train_data,
        test_data = test_data,
        epochs = epochs,
        batchsize = batchsize,
        learning_rate = 5e-4,
        accum_iter = 1,
        dev = dev
    )

    torch.save(model.state_dict(),
               'fb15k237_rgcn_{}_layers-{}_{}-{}_epochs.pt'.format(
                   rgcn_conf['n_layers'], rgcn_conf['rel_regularizer'], rgcn_conf['num_bases'], epochs)
               )

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
        graph_out, text_out = model(batch['entities'], batch['captions'])
        original_points.append(graph_encoder(batch['entities']).detach().cpu())
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

