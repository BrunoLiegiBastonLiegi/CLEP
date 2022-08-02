import torch, argparse, json, random, time, pickle
from dataset import CLIPDataset
from model import CLIP_KB, PretrainedGraphEncoder, GPT2CaptionEncoder, BertCaptionEncoder
from transformers import GPT2Tokenizer, BertTokenizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--train_data', help='Path to train data file.')
parser.add_argument('--test_data', help='Path to test data file.')
parser.add_argument('--graph_embeddings', default='data/pretrained_graph_embeddings.pkl', help='Path to the pretrained graph embedding file.')
parser.add_argument('--load_model', default=None, help='Path to caption pretrained model.')
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
with open('data/wid2idx_small.json', 'r') as f:
    wid2idx = json.load(f)
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
#valid_data = CLIPDataset(
#    datafile = args.valid_data, #'data/FB15k-237/valid.pkl',
#    tokenizer = tokenizer,
#    entity2idx = wid2idx,
#    device = dev
#)
print('> Initializing the model.')
# Graph encoder
with open(args.graph_embeddings, 'rb') as f:
    node_embeddings = pickle.load(f)
graph_encoder = PretrainedGraphEncoder(node_embeddings=node_embeddings, device=dev)
# Caption encoder
text_encoder = GPT2CaptionEncoder(pretrained_model='gpt2')
#text_encoder = BertCaptionEncoder(pretrained_model='bert-base-cased')
# CLIP
model = CLIP_KB(graph_encoder=graph_encoder, text_encoder=text_encoder, hdim=200).to(dev)

# Training
from utils import training_routine

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
    batchsize = 128
    
    training_routine(
        model = model,
        step_f = step_f,
        train_data = train_data,
        test_data = test_data,
        epochs = epochs,
        batchsize = batchsize,
        accum_iter = 1,
        dev = dev
    )

    torch.save(model.state_dict(), 'tmp.pt')

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
    distance, off_diag_dist = [], []
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
        distance.append(((graph_out-text_out)**2).sum(-1).sqrt())
        # Distance of offdiagonal pairs
        idx = set(range(graph_out.shape[0]))
        j = random.choices(list(idx), k=batchsize) # randomly sample a subset of offdiagonal pairs
        k = list(map(lambda x: random.choice(list(idx-{x})), j))
        off_diag_dist.append(((graph_out[j]-text_out[k])**2).sum(-1).sqrt())
        off_diag_dist.append(((graph_out[k]-text_out[j])**2).sum(-1).sqrt()) # asymettric in principle
        # Accuracy
        logits = torch.tensordot(graph_out, text_out.T, dims=1) #* torch.exp(model.T)
        for i, v in enumerate(sm(logits)):
            tot += 1
            if torch.argmax(v) == i:
                acc += 1
    distance = torch.cat(distance).detach().cpu().numpy()
    off_diag_dist = torch.cat(off_diag_dist).detach().cpu().numpy()
    print(ttest_ind(distance, off_diag_dist, equal_var=False))
    print(mannwhitneyu(distance, off_diag_dist))
    print(mannwhitneyu(distance, off_diag_dist, alternative='less'))
    print(mannwhitneyu(distance, off_diag_dist, alternative='greater'))
    #print(sorted(distance))
    ax.hist(distance, bins='auto', alpha=0.5, density=True)
    ax.hist(off_diag_dist, bins='auto', alpha=0.5, density=True)
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

