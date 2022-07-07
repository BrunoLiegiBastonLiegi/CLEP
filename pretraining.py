import torch, argparse, json, random, time, pickle
from dataset import CLIPDataset
from model import CLIP_KB, PretrainedGraphEncoder, GPT2CaptionEncoder
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

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
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side, tokenizer.pad_token = 'left', tokenizer.bos_token
print('> Preparing the data.')
# Load index mapping
#with open('data/FB15k-23/wid2idx_new.json', 'r') as f:
with open('data/wid2idx.json', 'r') as f:
    wid2idx = json.load(f)
print(len(wid2idx))
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
print(f'{len(train_data.data)} + {len(test_data.data)} = {len(train_data.data) + len(test_data.data)}')
# Graph encoder
#idx2emb = {wid2idx[d['wikidata_id']]: torch.tensor(d['embedding'], dtype=float) for d in train_data.data + test_data.data + valid_data.data}
#node_embeddings = dict(zip(list(wid2idx.values()), torch.zeros(len(wid2idx), 200)))
#idx2emb = {wid2idx[d['wikidata_id']]: torch.tensor(d['embedding'], dtype=float) for d in train_data.data + test_data.data}
#node_embeddings.update(idx2emb)
#del idx2emb
with open(args.graph_embeddings, 'rb') as f:
    node_embeddings = pickle.load(f)
print(len(node_embeddings))
graph_encoder = PretrainedGraphEncoder(node_embeddings=node_embeddings, device=dev)
# Caption encoder
text_encoder = GPT2CaptionEncoder(pretrained_model='gpt2')
# CLIP
model = CLIP_KB(graph_encoder=graph_encoder, text_encoder=text_encoder, hdim=200).to(dev)

def training_routine(model: CLIP_KB, train_data: CLIPDataset, test_data: CLIPDataset, device: torch.device = torch.device('cpu')):

    epochs = 32
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
        print(f'### EPOCH {e}')
        running_loss = 0.
        model.train()
        for i, (batch, label) in enumerate(train_loader):
            optimizer.zero_grad()
            with autocast():
                graph_out, text_out = model(batch['entities'], batch['captions'])
                logits = torch.tensordot(graph_out, text_out.T, dims=1) * torch.exp(model.T)
                if torch.isnan(graph_out).any() or torch.isnan(text_out).any():
                    print('isnan(graph): ',torch.isnan(graph_out).any())
                    print('isnan(text): ',torch.isnan(text_out).any())
                del graph_out
                del text_out
                torch.cuda.empty_cache()
                loss = 0.5 * ( loss_f(logits, label) + loss_f(logits.T, label) )
                if torch.isnan(loss):
                    print('isnan(logits): ', torch.isnan(logits).any())
                    print('e^t: ', torch.exp(model.T))
                del logits
                torch.cuda.empty_cache()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
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

time.sleep(10)
training_routine(model, train_data, test_data, device = dev)
#torch.save(model.state_dict(), 'model.pt')

#model.load_state_dict(torch.load('model.pt'))

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
    fig, ax = plt.subplots(1,1)
    model.eval()
    for batch, label in test_loader:
        graph_out, text_out = model(batch['entities'], batch['captions'])
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

    
