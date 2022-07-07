import argparse, torch, json
from dataset import LinkPredictionDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--train_data', help='Path to train data file.')
parser.add_argument('--test_data', help='Path to test data file.')
#parser.add_argument('--valid_data', help='Path to validation data file.')
#parser.add_argument('--entity_index', help='Path to index data file.')
parser.add_argument('--graph_embeddings', help='Path to pretrained embeddings file.')
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
    
# Train and Test data
train_data = LinkPredictionDataset(
    datafile = args.train_data, 
    entity2idx = wid2idx,
    predict = 'relation'
)
test_data = LinkPredictionDataset(
    datafile = args.test_data,
    entity2idx = wid2idx,
    predict = 'relation'
)

batchsize = 5
train_loader = DataLoader(
        train_data,
        batch_size = batchsize,
        shuffle = True,
        collate_fn = train_data.collate_fn
    )

for batch in train_loader:
    print(batch)

