import argparse, json, torch

from model import EntityLinkingModel
from utils import KG


if __name__ == "__main__":

    # Set device for computation
    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')
    print(f'\n> Setting device {dev} for computation.')

    # load the entity id map 
    with open("data/wikidata-disambig-cut/ent2idx.json", "r") as f:
        ent2idx = json.load(f)
    # load the relation id map 
    with open("data/wikidata-disambig-cut/rel2idx.json", "r") as f:
        rel2idx = json.load(f)

    # load the kg
    kg = KG(ent2idx=ent2idx, rel2idx=rel2idx, embedding_dim=200, dev=dev, add_inverse_edges=True)
    kg.build_from_file("data/wikidata-disambig-cut/link-prediction/train.txt")

    # prepare the graph encoder
    graph_encoder = "RGCN"
    if graph_encoderl == 'RGCN':
        conf = {
            'kg': kg,
            'n_layers': 2,
            'indim': kg.embedding_dim,
            'hdim': 200,
            'rel_regularizer': 'basis',
            'num_bases': 64
        }   
        graph_encoder = RGCN(**conf)
    elif graph_encoder == 'CompGCN':
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
    text_encoder = "gpt2"
    text_encoder = GPT2CaptionEncoder(pretrained_model=text_encoder)

    # load
    clep_model = CLIP_KB(
        graph_encoder = graph_encoder,
        text_encoder = text_encoder,
        hdim = 200
    ).to(dev)
    clep_model.load_state_dict(torch.load(args.load_model))

    # load the test data
    path = "data/wikidata-disambig-cut/entity-linking/wikidata-disambig-test.json"
    with open(path, 'r') as f:
        data = json.load(f)
        entity_mentions, entity_labels, entity_ids = [], [], []
        for d in data:
            entity_mentions.append(d["text"])
            entity_labels.append(d["string"])
            entity_ids.append(d["correct_id"])
    
    # evaluate the model on the test data
    hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    predictions, labels = [], []
    for mention, label, _id in zip(entity_mentions, entity_labels, entity_ids):
        groundtruth = ent2idx[_id]
        candidates = clep_model(mention, entity, top_k=max(hits_at_k))
        for i in hits_at_k.keys():
            if groundtruth in candidates[:i]:
                hits_at_k[i] += 1
        labels.append(groundtruth)
        predictions.append(candidates[0])
            
    
            

    

    
