import argparse, torch, json, pickle, time, random
from dataset import LinkPredictionDataset
from model import LinkPredictionModel, PretrainedGraphEncoder, MLP, CLIP_KB, GPT2CaptionEncoder, BertCaptionEncoder, ConcatModel, RGCN
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics import F1Score
from utils import training_routine, KG
from multiprocessing import Pool
from itertools import repeat
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='Caption prediction pretraining.')
parser.add_argument('--train_data', help='Path to train data file.')
parser.add_argument('--test_data', help='Path to test data file.')
parser.add_argument('--graph_embeddings', help='Path to pretrained embeddings file.')
parser.add_argument('--entity_index', default=None, help='Path to entity index file.')
parser.add_argument('--rel_index', help='Path to relations index file.')
parser.add_argument('--load_model', help='Path to caption pretrained model.')
parser.add_argument('--graph', default=None, help='Path to graph triples file.')
args = parser.parse_args()

# Set device for computation
if torch.cuda.is_available():
    dev = torch.device('cuda:0')
else:
    dev = torch.device('cpu')
print(f'\n> Setting device {dev} for computation.')

# Load index
with open(args.entity_index, 'r') as f:
    wid2idx = json.load(f)
with open (args.rel_index, 'r') as f:
    rel2idx = json.load(f)

# Train and Test data
train_data = LinkPredictionDataset(
    datafile = args.train_data, 
    entity2idx = wid2idx,
    rel2idx = rel2idx,
    #add_inverse_edges = True
)
test_data = LinkPredictionDataset(
    datafile = args.test_data,
    entity2idx = wid2idx,
    rel2idx = rel2idx,
    #add_inverse_edges = True
)
val_data = LinkPredictionDataset(
    datafile = 'data/FB15k-237/link-prediction/_valid_wiki-id.txt',
    entity2idx = wid2idx,
    rel2idx = rel2idx,
    #add_inverse_edges = True
)

rel2idx = train_data.r2idx

w = 2 # number of corrupted triples per positive triple 
train_data.generate_corrupted_triples(torch.cat([test_data.true_triples, val_data.true_triples]), mode='gen', w=w/2)
#train_data.generate_corrupted_triples('data/FB15k-237/corrupted_train_triples.pt', mode='load')
test_data.generate_corrupted_triples(torch.cat([train_data.true_triples, val_data.true_triples]), mode='gen', w=w/2)

if args.graph_embeddings != None:
    with open(args.graph_embeddings, 'rb') as f:
        node_embeddings = pickle.load(f)
# Baseline: pretrained TransE embeddings
#BaselineModel = PretrainedGraphEncoder(node_embeddings=node_embeddings, index=wid2idx, device=dev)

if  args.graph != None:
    #kg = KG(embedding_dim = 200, dev=dev)
    kg = KG(triples = train_data.true_triples, embedding_dim = 500, dev=dev)
    #kg.build_from_file(args.graph, wid2idx, rel2idx)
    #kg.node_feat = torch.load('data/FB15k-237/rgcn_initial_node_features.pt')

BaselineModel = RGCN(
    kg = kg,
    n_layers = 2,
    indim = kg.embedding_dim,
    hdim = 500,
    #rel_regularizer = 'basis',
    rel_regularizer = 'bdd',
    num_bases = 100,
    regularization = torch.nn.Dropout(0.2)
)

# Caption prediction pretraining
# Annoyingly I have to load the gpt model to load the weights I need, even though I am not
# going to use that. A possible solution would be to save the complete model instead of saving
# just the state_dict, that would require more disk space though.
# REMEMBER: I NEED TO RERUN PRETRAINING SINCE wid2idx.json HAS CHANGED
"""
_ = GPT2CaptionEncoder(pretrained_model='gpt2')
#_ = BertCaptionEncoder(pretrained_model='bert-base-cased')
clip = CLIP_KB(
    graph_encoder = RGCN(
        kg = kg,
        n_layers = 3,
        indim = kg.embedding_dim,
        hdim = 200,
        num_bases = 64
    ),
    text_encoder = _,
    hdim = 200
).to(dev)
clip.load_state_dict(torch.load(args.load_model))
# Stack the pretrained MLP on top of the graph embedding model
SemanticAugmentedModel = torch.nn.Sequential(clip.g_encoder, clip.g_mlp)
"""
#for par in SemanticAugmentedModel.parameters():
#    par.requires_grad = False
#ConcatModel = ConcatModel(BaselineModel, SemanticAugmentedModel)


# Training
# Define training step
def step_f(model, batch, label, dev):
    batch, label = batch.to(dev), label.to(dev)
    out = model(batch[:,0], batch[:,2], r=batch[:,1])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out,
        label,
        weight = torch.ones(batch.shape[0], device=dev)*1/(1+w) # w=2 ratio negative/positive triples (c = 1/(w+1))
    )
    del out, batch, label
    torch.cuda.empty_cache()
    return loss

def experiment(model, train_data, test_data, dev=dev, rel2idx=rel2idx):
    # build LP model
    LPmodel = LinkPredictionModel(
        graph_embedding_model = model,
        mode = 'Distmult',
        #mode = 'TransE',
        predict = 'head',
        rel2idx = rel2idx
    ).to(dev)
    # train
    epochs = 10
    #batchsize = 128
    batchsize = 8192
    lr = 1e-2
    training_routine(
        model = LPmodel,
        step_f = step_f,
        train_data = train_data,
        test_data = test_data,
        epochs = epochs,
        batchsize = batchsize,
        learning_rate = lr,
        accum_iter = 1,
        dev = dev
    )
    # test inference
    LPmodel.eval()
    #test_data.generate_evaluation_triples(triples=train_data.true_triples, corrupt='tail', mode='gen')
    test_data.triples = test_data.true_triples
    #print(len(test_data))
    test_loader = DataLoader(
        test_data,
        batch_size = 64,#128,#8192,
        shuffle = True,
        collate_fn = test_data.collate_fn
    )

    nodes = kg.g.nodes().detach().cpu()
    check_triples = torch.vstack((train_data.true_triples[:,:3], test_data.true_triples[:,:3], val_data.true_triples[:,:3]))
    check_triples = torch.vstack((check_triples, check_triples[:,[2,1,0]])) # do the same also for corrupted train triples
    check_triples = {tuple(t.tolist()) for t in check_triples}
    ranks = []

    el = 2 # head and/or tail
    el_slice = (1,3) if el == 0 else (0,2)
    for i, (batch, _) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            size = batch.shape[0]
            """ 
            for j in range(size): # Stupid loop to check that evaluation is right
                tmp_b = [batch[j].tolist()]
                for n in nodes:
                    tmp = batch[j,:2].clone().tolist() + [n]
                    if tuple(tmp) not in check_triples:
                        tmp_b.append(tmp)
                tmp_b = torch.as_tensor(tmp_b).to(dev)
                tmp_o = LPmodel(tmp_b[:,0], tmp_b[:,2], r=tmp_b[:,1]).detach().cpu()
                print((tmp_o.sort(descending=True).indices == 0).nonzero())
            """        
            # Generate the new corrupted heads
            new_el = torch.vstack([nodes for j in range(size)]).T.reshape(-1,1)
            # Copy the true batch in order to make place for the new heads
            corr_batch = torch.vstack([batch for j in range(nodes.shape[0])])
            # Attach the new heads
            corr_batch[:,el] = new_el[:,0]
            del new_el
            # Keep only the triples appearing in check triples
            # This is done by converting the tensor in a set object and using
            # a combination of symmetric difference and intersection
            corr_batch = set(zip(corr_batch[:,0].tolist(), corr_batch[:,1].tolist(), corr_batch[:,2].tolist()))
            corr_batch = torch.as_tensor(list((corr_batch.symmetric_difference(check_triples)).intersection(corr_batch)))
            #label = torch.cat([torch.ones(size), torch.zeros(corr_batch.shape[0])])
            corr_batch = torch.vstack((batch, corr_batch))
            corr_batch = corr_batch.to(dev)
            #out = torch.sigmoid(LPmodel(corr_batch[:,0], corr_batch[:,2], r=corr_batch[:,1])).detach().cpu()
            out = LPmodel(corr_batch[:,0], corr_batch[:,2], r=corr_batch[:,1]).detach().cpu()
            """
            for j,t in enumerate(corr_batch[:size]):
                if not (t.detach().cpu() == batch[j]).all():
                    print('> Different triples:', t, batch[j])
                    assert False
                mask = torch.cat(( 
                    torch.nn.functional.one_hot(torch.tensor(j, device=dev), num_classes=size).bool(), 
                    (corr_batch[size:,el_slice[0]:el_slice[1]] == t[el_slice[0]:el_slice[1]]).all(-1) 
                ), dim=0)
                print('> Mask:\n', mask.nonzero())
                rand_idx = random.choice(mask.nonzero())
                scores = out[mask]
                tmp = torch.as_tensor(list(check_triples))
                print('> Rand Score: ', corr_batch[rand_idx].tolist(), out[rand_idx], (corr_batch[rand_idx].cpu()==tmp).all(-1).nonzero().tolist())
                print('> True Score: ', 0, corr_batch[mask][0].tolist(), scores[0], (corr_batch[mask][0].cpu()==tmp).all(-1).nonzero().tolist())
                max_idx = scores.argmax()
                print(f'> Max Score: ', max_idx.item(), corr_batch[mask][max_idx].tolist(), scores[max_idx], (corr_batch[mask][max_idx].cpu()==tmp).all(-1).nonzero().tolist())
                tmp_ranks = scores.sort(descending=True)
                print(tmp_ranks.indices[:10], (tmp_ranks.indices==0).nonzero())
                del mask, scores, rand_idx
            """
            f = lambda x : (out[ # take only the scores of the relevant triples, i.e. those with the same head-rel or rel-tail
                torch.cat(( # concatenate the true triple (pos 0) and all the corrupted ones
                    torch.nn.functional.one_hot(torch.tensor(x[0], device=dev), num_classes=size).bool(), # true triple mask
                    (corr_batch[size:,el_slice[0]:el_slice[1]] == x[1][el_slice[0]:el_slice[1]]).all(-1) # relevant corrupted triples mask
                ), dim=0)
            ].sort(descending=True).indices == 0).nonzero() # sort the scores and look for the position of the true triple
            
            ranks.append(torch.vstack(list(map(
                f,
                enumerate(corr_batch[:size])
            ))).flatten())
            
            #for j, tt in enumerate(corr_batch[:size]):
            #    print(j)
            #    print(out.shape)
            #    print((corr_batch[size:,1:3] == tt[1:3]).all(-1).nonzero().shape)
                #print(torch.zeros(size).bool().shape)
            #    o = out[torch.cat((
            #        torch.nn.functional.one_hot(torch.tensor(j, device=dev), num_classes=size).bool(),
            #        (corr_batch[size:,1:3] == tt[1:3]).all(-1)),
            #                         dim=0)]
            #    print(o.shape)
            #print(time.time()-t)
    del nodes, check_triples
    ranks = torch.cat(ranks).view(-1) + 1 # +1 since the position starts counting from zero
    
    print(f'> Hits @10: {len((ranks <= 10).nonzero()) / len(ranks)}')
    print(f'> Mean Rank: {ranks.mean(dtype=float)}')
    MRR = (1/ranks).mean(dtype=float)
    print(f'> MRR: {MRR}')
    return MRR


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

#embs = get_embeddings(SemanticAugmentedModel, test_loader)
#clusters = visualize_embeddings(torch.vstack(list(embs.values())), n_clusters=50, ax=ax[0])

# Finetuning
outcomes = []
for m in (BaselineModel, ):
    scores = []
    for j in range(1):
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
    #outcomes.append({'micro F1': scores[:,0].mean(), 'macro F1': scores[:,1].mean()})

#embs = get_embeddings(SemanticAugmentedModel, test_loader)
#visualize_embeddings(torch.vstack(list(embs.values())), ax=ax[1])
#plt.show()

for o in outcomes:
    print(o)

