import argparse, json, torch
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from tqdm import tqdm

from model import EntityLinkingModel, CLIP_KB, PretrainedGraphEncoder, GPT2CaptionEncoder, CaptionEncoder, RGCN, CompGCNWrapper
from transformers import AutoTokenizer
from utils import KG


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Entity Linking.')
    parser.add_argument('--dataset')
    parser.add_argument('--load_model')
    parser.add_argument('--graph_encoder', default="RGCN")
    parser.add_argument('--text_encoder', default="gpt2")
    
    args = parser.parse_args()

    if args.dataset is not None:
        entity_index = 'data/{}/ent2idx.json'.format(args.dataset)
        rel_index = 'data/{}/rel2idx.json'.format(args.dataset)
        graph = 'data/{}/link-prediction/train.txt'.format(args.dataset)
        test_data = 'data/{}/entity-linking/test.json'.format(args.dataset)

    # Set device for computation
    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        try:
            dev = torch.device('mps')
        except RuntimeError:
            dev = torch.device('cpu')
    print(f'\n> Setting device {dev} for computation.')

    # load the entity id map 
    with open(entity_index, "r") as f:
        ent2idx = json.load(f)
    # load the relation id map 
    with open(rel_index, "r") as f:
        rel2idx = json.load(f)

    # load the kg
    kg = KG(ent2idx=ent2idx, rel2idx=rel2idx, embedding_dim=200, dev=dev, add_inverse_edges=True)
    kg.build_from_file(graph)

    # prepare the graph encoder
    if args.graph_encoder == 'RGCN':
        conf = {
            'kg': kg,
            'n_layers': 2,
            'indim': kg.embedding_dim,
            'hdim': 200,
            'rel_regularizer': 'basis',
            'num_bases': 64
        }   
        graph_encoder = RGCN(**conf)
    elif args.graph_encoder == 'CompGCN':
        conf = {
            'kg': kg,
            'n_layers': 2,
            'indim': kg.embedding_dim,
            'hdim': 200,
            'comp_fn': 'sub',
            'num_bases': -1,
            'return_rel_embs' : False
        }   
        graph_encoder = CompGCNWrapper(**conf)

    # load the CLEP pretrained model
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    if "gpt2" in args.text_encoder:
        text_encoder = GPT2CaptionEncoder(pretrained_model=args.text_encoder)
    else:
        text_encoder = CaptionEncoder(pretrained_model=args.text_encoder)

    # load
    clep_model = CLIP_KB(
        graph_encoder = graph_encoder,
        text_encoder = text_encoder,
        hdim = 200
    ).to(dev)
    clep_model.load_state_dict(torch.load(args.load_model))

    EL_model = EntityLinkingModel(clep_model, tokenizer) 

    # load the test data
    with open(test_data, 'r') as f:
        data = json.load(f)
        entity_mentions, entity_labels, entity_ids = [], [], []
        for d in data:
            entity_mentions.append(d["text"])
            entity_labels.append(d["string"])
            entity_ids.append(d["correct_id"])
    
    # evaluate the model on the test data
    hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0, 50: 0, 100: 0, 500: 0, 1000: 0, 5000: 0, 10000: 0}
    predictions, labels = [], []
    for mention, label, _id in tqdm(list(zip(entity_mentions, entity_labels, entity_ids)), total=len(entity_mentions)):
        groundtruth = ent2idx[_id]
        candidates = EL_model(mention, label, top_k=max(hits_at_k))
        for i in hits_at_k.keys():
            if groundtruth in candidates[:i]:
                hits_at_k[i] += 1
        labels.append(groundtruth)
        predictions.append(candidates[0])

    print(f"--> hits@k: {hits_at_k}")
    #print(f"--> F1 score: {f1_score(labels.cpu(), predictions.cpu())}")

    x, y = zip(*hits_at_k.items())
    plt.plot(x, y)
    plt.show()
    
            

    

    
# Notes:

# I tested this on the wikidata-disambig dataset with a CLEP(RGCN, gpt2) model trained with batch size 128
# but the results are poor:

# --> hits@k: {1: 2, 3: 3, 5: 4, 10: 11, 50: 36, 100: 71, 500: 346} (out of 10.000 test samples)

# The reasons could be several:
# - too many descriptions are missing, out of the roughly 80.000 entities, around 10.000 of them
#   have missing or completely uninformative descriptions, e.g. `wikidata disambiguation page`
# - the descriptions found in Wikidata are not particularly informative, with often a large degree of overlapping
#   among different entities, e.g. `american actress`, `rock band`, `capital city`. They might be ok for
#   some general identification of their type, but not precise enough to enable entity disambiguation
# - the graph is rather sparse and disconnected, only roughly 6.000 edges were found among 80.000 entities

# Possible ideas to explore:
# - use more informative descriptions for the pretraining, for instance by taking the first sentence of
#   corresponding wikipedia webpage to each entity, or asking an LLM to generate it
# - adapt another dataset for which we do have a comprehensive graph, e.g. FB15k-237, to the entity linking task.
#   For example, by taking any sentence in the wikipedia web page of an entity that contains it as the entity
#   mention to link to the KB.


# even working with the cut dataset, i.e. discarding the entities that miss descriptions, the entity linking
# performance does not get better:

# --> hits@k: {1: 2, 3: 3, 5: 5, 10: 12, 50: 51, 100: 88, 500: 328}

# with cosine similarity
# --> hits@k: {1: 1, 3: 1, 5: 3, 10: 6, 50: 20, 100: 38, 500: 186, 1000: 336, 5000: 1430, 10000: 2709}

# by using Minilm instead of gpt2 I was able to train with batchsize=10,000 on 16GB and batchsize=30,000 on 48GB
# this drastically reduce the problem of overfitting and leads to a way better alignment, even though the
# overlapping is still significant.

# candidates generation with these models improves significantly
# --> hits@k: {1: 24, 3: 48, 5: 70, 10: 129, 50: 417, 100: 653, 500: 1609, 1000: 2239, 5000: 4336, 10000: 5486}

# furthermore not normalizing the vectors seems to slightly help
# --> hits@k: {1: 31, 3: 71, 5: 125, 10: 221, 50: 753, 100: 1163, 500: 2434, 1000: 3060, 5000: 4861, 10000: 5672}
# with cosine similarity but not with l2 norm
