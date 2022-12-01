from torch.utils.data import Dataset
import pickle, torch, json, random
from tqdm import tqdm
from scipy.sparse import coo_matrix
from multiprocessing import Pool
from itertools import repeat
from dgl.sampling import global_uniform_negative_sampling


class CLIPDataset(Dataset):

    def __init__(self, datafile: str, tokenizer, entity2idx: dict, triples = None, device: torch.device = torch.device('cpu')):
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
            with open(datafile, 'r') as f:
                idx2cap = {
                    entity2idx[v['wikidata_id']]: v['caption']
                    for v in json.load(f).values()
                    }
            self.data = []
            for t in triples:
                cap = idx2cap[t[2].item()]
                if cap is not None:
                    self.data.append((t[:2], cap))
                else:
                    self.data.append((t[:2], 'Caption not available.'))
        self.tok = tokenizer
        self.e2idx = entity2idx
        self.dev = device
        #self.discard_incomplete()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def discard_incomplete(self):
        incomplete_idx = []
        for i,d in enumerate(self.data):
            flag = False
            if isinstance(d,dict):
                eid = d['wikidata_id'] if 'wikidata_id' in d.keys() else d['entity_id']
                if eid == None or d['caption'] == None:
                    flag = True
            else:
                flag = True
            incomplete_idx.append(flag)
            #if flag:
            #    try:
            #        self.e2idx.pop(d['wikidata_id'])
            #    except:
            #        continue
        self.data = [ d for d,f in zip(self.data, incomplete_idx) if not f ]
        #self.e2idx = dict(zip(self.e2idx.keys(), range(len(self.e2idx))))

    def collate_fn(self, batch: list):
        inputs = {'captions':[], 'entities':[]}
        for item in batch:
            if self.h_to_t:
                inputs['captions'].append(item[1])
                inputs['entities'].append(item[0])
            else:
                inputs['captions'].append(item['caption'])
                eid = item['wikidata_id'] if 'wikidata_id' in item.keys() else item['entity_id']
                inputs['entities'].append(self.e2idx[eid])
        inputs['captions'] = self.tok(text=inputs['captions'], padding=True, return_tensors='pt')#.to(self.dev)
        #inputs['entities'] = torch.as_tensor(inputs['entities'])#.to(self.dev)
        inputs['entities'] = torch.vstack(inputs['entities']) if self.h_to_t else torch.as_tensor(inputs['entities'])
        labels = torch.arange(len(batch))#.to(self.dev)
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

    def generate_corrupted_triples(self, triples : torch.tensor = None, mode: str ='gen', pos : list = ['head','tail'], w: int = 1):
        assert mode in ('gen', 'load')
        if mode == 'gen':
            pos2idx = {'head':0, 'tail':2}
            print('> Generating corrupted triples ...')
            pos = [pos2idx[p] for p in pos]
            if self.inv_triples == None:
                corrupted_triples = torch.vstack([ self._corrupt_triple(t, triples, position=pos, w=w) for t in tqdm(self.true_triples[:,:3]) ])
            else:
                corrupted_triples = torch.vstack([ self._corrupt_triple(t, triples, position=pos, w=w) for t in tqdm(torch.vstack((self.true_triples[:,:3], self.inv_triples[:,:3]))) ])
            name = input('Save corrupted triples to:\n\t')
            torch.save(corrupted_triples, name)
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

    def generate_evaluation_triples(self, triples: torch.tensor = None, corrupt: str = 'tail', mode: str ='gen'):
        """Takes too much memory.
        """
        assert corrupt in ('head', 'relation', 'tail')
        corr_idx = {'head': 0, 'relation': 1, 'tail': 2}
        nodes = torch.tensor(list(self.e2idx.values()))
        # correct triples used for checking
        check_triples = torch.vstack((self.true_triples[:,:3], triples[:,:3])) if triples != None else self.true_triples[:,:3]
        # new head/rel/tail
        print('Generating new_el')
        new_el = torch.vstack([nodes for j in range(self.true_triples.shape[0])]).T.reshape(-1,1)
        # corrupted triples
        print('Generating corrupted triples')
        corr_t = torch.vstack([self.true_triples[:,:3] for j in range(nodes.shape[0])]) # copy and repeat the correct triples
        corr_t[:,corr_idx[corrupt]] = new_el[:,0] # corrupt the correct triples by injecting the new elements
        del nodes, new_el
        corr_t = set(zip(corr_t[:,0].tolist(), corr_t[:,1].tolist(), corr_t[:,2].tolist()))
        print(f'> Intersection with check_triples')
        corr_t = torch.as_tensor(list((corr_t.symmetric_difference(check_triples)).intersection(corr_t)))
        # check which of the corrupted triples don't appear in the corret ones
        #for t in tqdm(corr_t):
        #    ((x!=check_triples).sum(-1) == 0).sum().bool()
        #idx = torch.vstack(list(map(
        #    lambda x: ((x!=check_triples).sum(-1) == 0).sum().bool(),
        #    corr_t
        #))).flatten()
        # keep only the corrupted triples not present in the correct ones
        #corr_t = corr_t[~idx,:]
        # eliminate repeated triples
        #corr_t = torch.unique(corr_t, dim=0)
    
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
        
