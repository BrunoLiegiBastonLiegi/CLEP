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
#parser.add_argument('--valid_data', help='Path to validation data file.')
parser.add_argument('--graph_embeddings', default='data/pretrained_graph_embeddings.pkl', help='Path to the pretrained graph embedding file.')
args = parser.parse_args()

# Set device for computation
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
else:
    dev = torch.device('cpu')
print(f'\n> Setting device {dev} for computation.')
# Choose the tokenizer
print(f'> Loading Pretrained tokenizer.')
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#tokenizer.padding_side, tokenizer.pad_token = 'left', tokenizer.bos_token
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
print('> Preparing the data.')
# Load index mapping
with open('data/wid2idx.json', 'r') as f:
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
#text_encoder = GPT2CaptionEncoder(pretrained_model='gpt2')
text_encoder = BertCaptionEncoder(pretrained_model='bert-base-cased')
# CLIP
model = CLIP_KB(graph_encoder=graph_encoder, text_encoder=text_encoder, hdim=200).to(dev)

def training_routine(model: CLIP_KB, train_data: CLIPDataset, test_data: CLIPDataset, accum_iter: int = 1, device: torch.device = torch.device('cpu')):

    epochs = 11
    batchsize = 128
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss_f = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    train_loader = DataLoader(
        train_data,
        batch_size = batchsize,
        shuffle = True,
        collate_fn = train_data.collate_fn
    )

    test_loader = DataLoader(
        test_data,
        batch_size = batchsize,
        shuffle = True,
        collate_fn = test_data.collate_fn
    )
    print_steps = int(len(train_loader)/5)
    for e in range(epochs):
        print(f'\n### EPOCH {e}')
        running_loss = 0.
        model.train()
        for i, (batch, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            #optimizer.zero_grad()
            with autocast():
                graph_out, text_out = model(batch['entities'], batch['captions'])
                logits = torch.tensordot(graph_out, text_out.T, dims=1) * torch.exp(model.T)
                del graph_out
                del text_out
                torch.cuda.empty_cache()
                loss = 0.5 * ( loss_f(logits, label) + loss_f(logits.T, label) )
                running_loss += loss.item()
                # normalize loss to account for batch accumulation
                loss = loss / accum_iter 
                del logits
                torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            # weights update
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if i % print_steps == print_steps - 1:
                print(f'[{e}, {i*batchsize}]\t Loss: {running_loss/print_steps:.4f}\t T: {model.T:0.3f} ') # The average is not correct if len(train_loader) % batchsize != 0
                running_loss = 0.
        running_loss = 0.
        model.eval()
        for i, (batch, label) in enumerate(test_loader):
            with torch.no_grad():
                graph_out, text_out = model(batch['entities'], batch['captions'])
                logits = torch.tensordot(graph_out, text_out.T, dims=1) * torch.exp(model.T)
                loss = 0.5 * ( loss_f(logits, label) + loss_f(logits.T, label) )
                running_loss += loss.item()
        print(f'> Test Loss: {running_loss/(len(test_loader)):.4f}')

#training_routine(model, train_data, test_data, accum_iter = 1, device = dev)
#torch.save(model.state_dict(), 'tmp.pt')

model.load_state_dict(torch.load('tmp.pt'))

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
    original_points,points = [], []
    fig, ax = plt.subplots(1,1)
    model.eval()
    for batch, label in test_loader:
        graph_out, text_out = model(batch['entities'], batch['captions'])
        original_points.append(graph_encoder(batch['entities']).detach().cpu())
        points.append(graph_out.detach().cpu()) # points for space visualization
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
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    fig, ax = plt.subplots(1,1, figsize=(16,16))
    points = torch.vstack(points).numpy()
    original_points = torch.vstack(original_points).numpy()
    n_clusters = 52 # 52 appears to be the optimal number
    sse, sil_coeff = {}, []
    #for n in range(2,200):
    #    print(f'k-means {n}/200', end='\r')
    #    #kmeans = KMeans(n_clusters=n, random_state=0).fit(points)
    #    kmeans = KMeans(n_clusters=n, random_state=0).fit(original_points)
    #    sse[n] = kmeans.inertia_
    #    #sil_coeff.append((n,silhouette_score(points, kmeans.labels_, metric='euclidean')))
    #    sil_coeff.append((n,silhouette_score(original_points, kmeans.labels_, metric='euclidean')))
    #sil_coeff = sorted(sil_coeff, key=lambda x: x[1], reverse=True)[:20]
    #for i in sil_coeff:
    #    print(i)
    #plt.plot(list(sse.keys()), list(sse.values()))
    #plt.show()
    #kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(points)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(original_points)
    clusters = {i:[] for i in range(n_clusters)}
    for j,i in enumerate(kmeans):
        clusters[i].append(j)
    transf = TSNE(n_components=2,init='random').fit_transform(points)
    for k, v in clusters.items():
        if len(v) != 0:
            ax.scatter(transf[v,0], transf[v,1])
    plt.show()
