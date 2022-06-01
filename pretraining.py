import torch, argparse, json
from dataset import CLIPDataset
from model import CLIP_KB, PretrainedGraphEncoder, GPT2CaptionEncoder
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Caption prediction pretraining.')

# Set device for computation
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
#dev = torch.device('cpu')
# Choose the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side, tokenizer.pad_token = 'left', tokenizer.bos_token
# Load index mapping
with open('data/FB15k-237/wid2idx.json', 'r') as f:
    wid2idx = json.load(f)
# Train and Test data
train_data = CLIPDataset(
    datafile = 'data/FB15k-237/train.pkl',
    tokenizer = tokenizer,
    entity2idx = wid2idx,
    device = dev
)
test_data = CLIPDataset(
    datafile = 'data/FB15k-237/test.pkl',
    tokenizer = tokenizer,
    entity2idx = wid2idx,
    device = dev
)
valid_data = CLIPDataset(
    datafile = 'data/FB15k-237/valid.pkl',
    tokenizer = tokenizer,
    entity2idx = wid2idx,
    device = dev
)
# Graph encoder
idx2emb = {wid2idx[d['wikidata_id']]: torch.tensor(d['embedding'], dtype=float) for d in train_data.data + test_data.data + valid_data.data}
graph_encoder = PretrainedGraphEncoder(node_embeddings=idx2emb)
# Caption encoder
text_encoder = GPT2CaptionEncoder(pretrained_model='gpt2')
# CLIP
model = CLIP_KB(graph_encoder=graph_encoder, text_encoder=text_encoder, hdim=200).to(dev)

def training_routine(model: CLIP_KB, train_data: CLIPDataset, test_data: CLIPDataset, device: torch.device = torch.device('cpu')):

    epochs = 32
    batchsize = 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    T = torch.nn.parameter.Parameter(torch.tensor(0.07).to(device), requires_grad = True) # Temperature
    loss_f = torch.nn.CrossEntropyLoss()

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
    for e in range(epochs):
        print(f'### EPOCH {e}')
        for batch, label in train_loader:
            optimizer.zero_grad()
            graph_out, text_out = model(batch['entities'], batch['captions'])
            if T >= 100:
                T = 100
            logits = torch.tensordot(graph_out, text_out.T, dims=1) * torch.exp(T)
            loss = 0.5 * ( loss_f(logits, label) + loss_f(logits.T, label) )
            print(loss,end='\r')
            loss.backward()
            optimizer.step()
        

training_routine(model, train_data, train_data, device = dev)
