import argparse, torch, json, pickle
from dataset import LinkPredictionDataset
from model import LinkPredictionModel, PretrainedGraphEncoder, MLP, CLIP_KB, GPT2CaptionEncoder, BertCaptionEncoder, ConcatModel
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics import F1Score

parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--train_data', help='Path to train data file.')
parser.add_argument('--test_data', help='Path to test data file.')
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
with open('data/wid2idx_small.json', 'r') as f:
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
_ = GPT2CaptionEncoder(pretrained_model='gpt2')
#_ = BertCaptionEncoder(pretrained_model='bert-base-cased')
clip = CLIP_KB(graph_encoder=BaselineModel, text_encoder=_, hdim=200).to(dev)
clip.load_state_dict(torch.load(args.load_model))
# Stack the pretrained MLP on top of the graph embedding model
SemanticAugmentedModel = torch.nn.Sequential(BaselineModel, clip.g_mlp)
for par in SemanticAugmentedModel.parameters():
    par.requires_grad = False
ConcatModel = ConcatModel(BaselineModel, SemanticAugmentedModel)


# Training
from utils import training_routine

# Define training step
def step_f(model, batch, label, dev):
    batch, label = batch.to(dev), label.to(dev)
    out = model(batch[:,0], batch[:,1])
    loss = torch.nn.functional.cross_entropy(out, label)
    del out, batch, label
    torch.cuda.empty_cache()
    return loss

def experiment(model, train_data, test_data, dev=dev, rel2idx=rel2idx):
    # build LP model
    LPmodel = LinkPredictionModel(
        graph_embedding_model = model,
        predict = 'relation',
        rel2idx = rel2idx
    ).to(dev)
    # train
    epochs = 16
    batchsize = 4096
    training_routine(
        model = LPmodel,
        step_f = step_f,
        train_data = train_data,
        test_data = test_data,
        epochs = epochs,
        batchsize = batchsize,
        accum_iter = 1,
        dev = dev
    )
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
    microf1 = F1Score(num_classes=len(rel2idx), average='micro').to(dev)
    macrof1 = F1Score(num_classes=len(rel2idx), average='macro').to(dev)
    for i, (batch, label) in enumerate(test_loader):
        with torch.no_grad():
            batch, label = batch.to(dev), label.to(dev)
            out = LPmodel(batch[:,0], batch[:,1])
            pred.append(torch.argmax(sm(out), dim=-1))
            target.append(label)

    pred = torch.cat(pred).view(-1)
    target = torch.cat(target).view(-1)
    return (microf1(pred, target), macrof1(pred, target))


# Latent space visualilzation
from utils import visualize_embeddings
import matplotlib.pyplot as plt
test_loader = DataLoader(
        test_data,
        batch_size = 8192,
        shuffle = False,
        collate_fn = test_data.collate_fn
    )

def get_embeddings(model, loader):
    embs = {}
    for i, (batch, _) in enumerate(loader):
        print(f'{i}/{len(test_loader)}', end='\r')
        with torch.no_grad():
            batch = batch.view(-1,1).to(dev)
            out = model(batch)
            embs.update(dict(zip(batch.flatten().detach().cpu().tolist(),out.detach().cpu())))
    return embs

fig, ax = plt.subplots(1,2, figsize=(24,16))

embs = get_embeddings(SemanticAugmentedModel, test_loader)
clusters = visualize_embeddings(torch.vstack(list(embs.values())), n_clusters=50, ax=ax[0])

# Finetuning
outcomes = []
for m in (SemanticAugmentedModel, ):
    scores = []
    for j in range(5):
        scores.append(
            experiment(
                model = m,
                train_data = train_data,
                test_data = test_data,
                dev = dev,
                rel2idx = rel2idx
            )
        )            
    scores = torch.as_tensor(scores)
    outcomes.append({'micro F1': scores[:,0].mean(), 'macro F1': scores[:,1].mean()})

embs = get_embeddings(SemanticAugmentedModel, test_loader)
visualize_embeddings(torch.vstack(list(embs.values())), ax=ax[1])
plt.show()

for o in outcomes:
    print(o)

