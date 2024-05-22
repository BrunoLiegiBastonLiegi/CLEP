import argparse, json, torch

from sklearn.metrics import f1_score

from model import EntityLinkingModel, CLIP_KB, PretrainedGraphEncoder, GPT2CaptionEncoder, BertCaptionEncoder, RGCN, CompGCNWrapper
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
    text_encoder = GPT2CaptionEncoder(pretrained_model=args.text_encoder)

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
    hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    predictions, labels = [], []
    for mention, label, _id in zip(entity_mentions, entity_labels, entity_ids):
        groundtruth = ent2idx[_id]
        candidates = EL_model(mention, label, top_k=max(hits_at_k))
        for i in hits_at_k.keys():
            if groundtruth in candidates[:i]:
                hits_at_k[i] += 1
        labels.append(groundtruth)
        predictions.append(candidates[0])

    print(f"--> F1 score: {f1_score(labels, predictions)}")
    
            

    

    
