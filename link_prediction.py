import argparse, torch, json, pickle
from dataset import LinkPredictionDataset
from model import LinkPredictionModel, PretrainedGraphEncoder, MLP, CLIP_KB, GPT2CaptionEncoder, BertCaptionEncoder
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics import F1Score

parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--train_data', help='Path to train data file.')
parser.add_argument('--test_data', help='Path to test data file.')
#parser.add_argument('--valid_data', help='Path to validation data file.')
#parser.add_argument('--entity_index', help='Path to index data file.')
parser.add_argument('--graph_embeddings', help='Path to pretrained embeddings file.')
parser.add_argument('--rel_index', help='Path to relations index file.')
parser.add_argument('--load_model', help='Path to caption pretrained model.')
args = parser.parse_args()


# Set device for computation
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
else:
    dev = torch.device('cpu')
print(f'\n> Setting device {dev} for computation.')

# Load index
with open('data/wid2idx.json', 'r') as f:
    wid2idx = json.load(f)
with open (args.rel_index, 'r') as f:
    rel2idx = json.load(f)
    
# Train and Test data
train_data = LinkPredictionDataset(
    datafile = args.train_data, 
    entity2idx = wid2idx,
    rel2idx = rel2idx,
    predict = 'relation'
)
test_data = LinkPredictionDataset(
    datafile = args.test_data,
    entity2idx = wid2idx,
    rel2idx = rel2idx,
    predict = 'relation'
)

with open(args.graph_embeddings, 'rb') as f:
    node_embeddings = pickle.load(f)
# Baseline: pretrained TransE embeddings
BaselineModel = PretrainedGraphEncoder(node_embeddings=node_embeddings, device=dev)
# Caption prediction pretraining
# Annoyingly I have to load the gpt model to load the weights I need, even though I am not
# going to use that. A possible solution would be to save the complete model instead of saving
# just the state_dict, that would require more disk space though.
#_ = GPT2CaptionEncoder(pretrained_model='gpt2')
_ = BertCaptionEncoder(pretrained_model='bert-base-cased')
clip = CLIP_KB(graph_encoder=BaselineModel, text_encoder=_, hdim=200).to(dev)
clip.load_state_dict(torch.load(args.load_model))
# Stack the pretrained MLP on top of the graph embedding model
SemanticAugmentedModel = torch.nn.Sequential(BaselineModel, clip.g_mlp)
for par in SemanticAugmentedModel.parameters():
    par.requires_grad = False

def training_routine(model: LinkPredictionModel, train_data: LinkPredictionDataset, test_data: LinkPredictionDataset, accum_iter: int = 1, device: torch.device = torch.device('cpu')):

    epochs = 10
    batchsize = 4096#1024
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
                batch, label = batch.to(dev), label.to(dev)
                out = model(batch[:,0], batch[:,1])
                loss = loss_f(out, label)
                del out, batch, label
                torch.cuda.empty_cache()
                running_loss += loss.item()
                # normalize loss to account for batch accumulation
                loss = loss / accum_iter 
            scaler.scale(loss).backward()
            # weights update
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if i % print_steps == print_steps - 1:
                print(f'[{e}, {i*batchsize}]\t Loss: {running_loss/print_steps:.4f}') # The average is not correct if len(train_loader) % batchsize != 0
                running_loss = 0.
        running_loss = 0.
        model.eval()
        for i, (batch, label) in enumerate(test_loader):
            with torch.no_grad():
                batch, label = batch.to(dev), label.to(dev)
                out = model(batch[:,0], batch[:,1])
                loss = loss_f(out, label)
                running_loss += loss.item()
        print(f'> Test Loss: {running_loss/(len(test_loader)):.4f}')

microf1 = F1Score(num_classes=len(rel2idx), average='micro').to(dev)
macrof1 = F1Score(num_classes=len(rel2idx), average='macro').to(dev)

for m in (SemanticAugmentedModel, BaselineModel):
    # build LP model
    LPmodel = LinkPredictionModel(
        graph_embedding_model = m,
        predict = 'relation',
        rel2idx = rel2idx
    ).to(dev)
    # train
    training_routine(LPmodel, train_data, test_data, accum_iter = 1, device = dev)
    # test inference
    LPmodel.eval()
    test_loader = DataLoader(
        test_data,
        batch_size = 8192,
        shuffle = True,
        collate_fn = test_data.collate_fn
    )
    sm = torch.nn.Softmax(1)
    pred, target = [], []
    for i, (batch, label) in enumerate(test_loader):
        with torch.no_grad():
            batch, label = batch.to(dev), label.to(dev)
            out = LPmodel(batch[:,0], batch[:,1])
            pred.append(torch.argmax(sm(out), dim=-1))
            target.append(label)

    pred = torch.cat(pred).view(-1)
    target = torch.cat(target).view(-1)
    print(microf1(pred, target), macrof1(pred, target))
    n=0
    for p,t in zip(pred,target):
        if p == t:
            n+=1
    print(f'{n}/{len(pred)} ({n/len(pred):0.4f})')
