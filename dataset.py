from torch.utils.data import Dataset
import pickle, torch, json, random
from tqdm import tqdm
from scipy.sparse import coo_matrix
from multiprocessing import Pool
from itertools import repeat
from dgl.sampling import global_uniform_negative_sampling


class CLIPDataset(Dataset):

    def __init__(self, datafile: str, tokenizer, entity2idx: dict, triples = None, filter_triples = None, device: torch.device = torch.device('cpu'), concatenate_labels=False):
        self.h_to_t = False
        if triples == None:
            if datafile[-4:] == '.pkl': 
                with open(datafile, 'rb') as f:
                    self.data = list(pickle.load(f).values())
            elif datafile[-5:] == '.json':
                with open(datafile, 'r') as f:
                    self.data = list(json.load(f).values())
            for d in self.data:
                if d['caption'] is None:
                    d['caption'] = 'Caption not available.'
        else:
            self.h_to_t = True
            assert filter_triples is not None
            self.filter_triples = filter_triples
            with open(datafile, 'r') as f:
                self.idx2cap = {
                    entity2idx[v['entity_id']]: v['caption']
                    for v in json.load(f).values()
                }
            self.data = []
            for t in triples:
                cap = self.idx2cap[t[2].item()]
                if cap is not None:
                    self.data.append((t, cap))
                else:
                    self.data.append((t, 'Caption not available.'))
        self.tok = tokenizer
        self.e2idx = entity2idx
        self.dev = device
        self.concatenate_labels = concatenate_labels
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def collate_fn(self, batch: list):
        inputs = {'captions':[], 'entities':[]}
        for item in batch:
            if self.h_to_t:
                if self.concatenate_labels:
                    raise NotImplementedError
                inputs['captions'].append(item[1])
                inputs['entities'].append(item[0])
            else:
                caption = item['caption']
                if self.concatenate_labels:
                    caption = f"{item['label']}: {caption}"
                inputs['captions'].append(caption)
                eid = item['wikidata_id'] if 'wikidata_id' in item.keys() else item['entity_id']
                inputs['entities'].append(self.e2idx[eid])
        inputs['captions'] = self.tok(text=inputs['captions'], padding=True, return_tensors='pt')#.to(self.dev)
        inputs['entities'] = torch.vstack(inputs['entities']) if self.h_to_t else torch.as_tensor(inputs['entities'])
        if self.h_to_t:
            head_mask = (inputs['entities'][:,[0,1]].view(-1,1,2).to(self.dev) == self.filter_triples[:,[0,1]]).all(-1)
            tail_mask = inputs['entities'][:,2].view(-1,1).to(self.dev) == self.filter_triples[:,2]
            labels = head_mask.float() @ tail_mask.T.float()
            inputs['entities'] = inputs['entities'][:,:2]
        else:
            labels = torch.arange(len(batch))
        return inputs, labels
        

class LinkPredictionDataset(Dataset):

    def __init__(self, datafile: str, entity2idx: dict, rel2idx, KG=None, add_inverse_edges: bool = False):
        self.e2idx = entity2idx
        self.r2idx = rel2idx.copy()
        self.kg = KG
        self.inv_triples = [] if add_inverse_edges else None
        if add_inverse_edges:
            self.r2idx.update({
                k+'^(-1)': v
                for k,v in zip(rel2idx.keys(), range(len(rel2idx), 2*len(rel2idx)))
            })
        self.discarded_triples = []
        with open(datafile, 'r') as f:
            self.triples = []
            for l in f:
                discard = False
                triple = l.split()
                try:
                    h, t = self.e2idx[triple[0]], self.e2idx[triple[2]]
                    r = self.r2idx[triple[1]]
                    #if h == 8235 or t == 8235 or h == 215 or t == 215: # entity 8235 seemed to cause nan errs in some cases
                        #assert False
                except:
                    discard = True
                    self.discarded_triples.append(l)
                if not discard:
                    self.triples.append(torch.as_tensor([h,r,t]))
                    if add_inverse_edges:
                        r = self.r2idx[triple[1]+'^(-1)'] 
                        self.inv_triples.append(torch.as_tensor([t,r,h]))
            self.triples = torch.vstack(self.triples)
            self.triples = torch.hstack((
                self.triples,
                torch.ones(self.triples.shape[0], 1, dtype=torch.int64)
            ))
            self.true_triples = self.triples.clone()
            if add_inverse_edges:
                self.inv_triples = torch.vstack(self.inv_triples)
                self.inv_triples = torch.hstack((
                    self.inv_triples,
                    torch.ones(self.inv_triples.shape[0], 1, dtype=torch.int64)
                ))
                self.triples = torch.vstack((
                    self.triples,
                    self.inv_triples
                ))

        print(f'> {len(self.discarded_triples)} discarded triples due to missing mapping in the index files.')

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int):
        return self.triples[idx]

    def generate_corrupted_triples(self, triples : torch.tensor = None, mode: str ='gen', pos : list = ['head','tail'], w: int = 1, save=None):
        assert mode in ('gen', 'load')
        if mode == 'gen':
            pos2idx = {'head':0, 'tail':2}
            print('> Generating corrupted triples ...')
            pos = [pos2idx[p] for p in pos]
            if self.inv_triples == None:
                corrupted_triples = torch.vstack([ self._corrupt_triple(t, triples, position=pos, w=w) for t in tqdm(self.true_triples[:,:3]) ])
            else:
                corrupted_triples = torch.vstack([ self._corrupt_triple(t, triples, position=pos, w=w) for t in tqdm(torch.vstack((self.true_triples[:,:3], self.inv_triples[:,:3]))) ])
            if save is not None:
                torch.save(corrupted_triples, save)
        elif mode == 'load':
            print(f'> Loading corrupted triples from {triples}.')
            corrupted_triples = torch.load(triples)
        corrupted_triples = torch.hstack((corrupted_triples, torch.zeros(corrupted_triples.shape[0], 1, dtype=torch.int64)))
        self.triples = torch.vstack((self.triples, corrupted_triples)).long()
        self.triples = self.triples[torch.randperm(len(self.triples))]
        
    def _corrupt_triple(self, t, filter_triples, val=None, position=[0,2], w=1):
        ct = []
        for n in range(w):     
            for i in position: # head/rel/tail
                while True:
                    corr_t = t.clone()
                    corr_t[i] = random.choice(list(self.e2idx.values())) if val == None else val
                    #if corr_t[0] == corr_t[2] and c == None: # discard self loops, i.e. triples of the form (u,r,u)
                    #    continue
                    if len((corr_t == filter_triples).all(-1).nonzero()) == 0:
                        ct.append(corr_t)
                        break
                    elif val != None:
                        break
        return torch.vstack(ct)
    
    def collate_fn(self, batch: list):
        t = torch.vstack(batch)
        triples, labels = t[:,:-1], t[:,-1].float()
        return triples, labels
            

class NodeClassificationDataset(Dataset):

    def __init__(self, datafile: str, entity2idx: dict, type2idx: dict):
        if datafile[-4:] == '.pkl': 
            with open(datafile, 'rb') as f:
                self.data = pickle.load(f)
        elif datafile[-5:] == '.json':
            with open(datafile, 'r') as f:
                self.data = json.load(f)
        self.data = list(self.data.values())
        self.e2idx = entity2idx
        self.t2idx = type2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def collate_fn(self, batch: list):
        inputs, labels = [], []
        for item in batch:
            inputs.append(self.e2idx[item['wikidata_id']])
            labels.append(self.t2idx[item['type']])
        return torch.as_tensor(inputs), torch.as_tensor(labels)
        
