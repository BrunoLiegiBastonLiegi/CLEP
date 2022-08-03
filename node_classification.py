import argparse, torch, json, pickle
from dataset import NodeClassificationDataset
from model import PretrainedGraphEncoder, MLP, CLIP_KB, GPT2CaptionEncoder, BertCaptionEncoder, ConcatModel
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics import F1Score
import matplotlib.pyplot as plt


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

# Load index
with open('data/wid2idx_small.json', 'r') as f:
    wid2idx = json.load(f)
type2idx = {'PER': 0, 'LOC': 1, 'ORG': 2}
    
# Train and Test data
train_data = NodeClassificationDataset(
    datafile = args.train_data, 
    entity2idx = wid2idx,
    type2idx = type2idx
)
test_data = NodeClassificationDataset(
    datafile = args.test_data,
    entity2idx = wid2idx,
    type2idx = type2idx
)

# Pretrained Graph Embeddings
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
#for par in SemanticAugmentedModel.parameters():
#    par.requires_grad = False

# Training
from utils import training_routine

# Define training step
def step_f(model, batch, label, dev):
    batch, label = batch.to(dev), label.to(dev)
    out = model(batch)
    loss = torch.nn.functional.cross_entropy(out, label)
    del out, batch, label
    torch.cuda.empty_cache()
    return loss

def experiment(model, train_data, test_data, dev=dev):

    NChead = MLP(
        n_layers = 2,
        indim = 200,
        hdim = 200,
        outdim = 3
    )
    NCmodel = torch.nn.Sequential(model, NChead).to(dev)

    epochs = 50#64
    batchsize = 4096
    train_loss, test_loss = training_routine(
        model = NCmodel,
        step_f = step_f,
        train_data = train_data,
        test_data = test_data,
        epochs = epochs,
        batchsize = batchsize,
        accum_iter = 1,
        dev = dev
    )
    #plt.plot(train_loss[1:])
    #plt.plot(test_loss[1:])
    #plt.show()
    # test inference
    NCmodel.eval()
    test_loader = DataLoader(
        test_data,
        batch_size = 8192,
        shuffle = True,
        collate_fn = test_data.collate_fn
    )
    sm = torch.nn.Softmax(1)
    pred, target = [], []
    microf1 = F1Score(num_classes=len(type2idx), average='micro').to(dev)
    macrof1 = F1Score(num_classes=len(type2idx), average='macro').to(dev)
    for i, (batch, label) in enumerate(test_loader):
        with torch.no_grad():
            batch, label = batch.to(dev), label.to(dev)
            out = NCmodel(batch)
            pred.append(torch.argmax(sm(out), dim=-1))
            target.append(label)

    pred = torch.cat(pred).view(-1)
    target = torch.cat(target).view(-1)
    return (microf1(pred, target), macrof1(pred, target))


outcomes = []
for m in (SemanticAugmentedModel, BaselineModel):
    scores = []
    for j in range(5):
        scores.append(
            experiment(
                model = m,
                train_data = train_data,
                test_data = test_data,
                dev = dev,
            )
        )            
    scores = torch.as_tensor(scores)
    outcomes.append({'micro F1': scores[:,0].mean(), 'macro F1': scores[:,1].mean()})

for o in outcomes:
    print(o)
