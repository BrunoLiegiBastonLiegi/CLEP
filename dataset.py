from torch.utils.data import Dataset
import pickle, torch, json


class CLIPDataset(Dataset):

    def __init__(self, datafile: str, tokenizer, entity2idx: dict, device: torch.device = torch.device('cpu')):
        with open(datafile, 'rb') as f:
            self.data = list(pickle.load(f).values())
            #self.data = list(json.load(f).values())
        self.tok = tokenizer
        self.e2idx = entity2idx
        self.dev = device
        self.discard_incomplete()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def discard_incomplete(self):
        incomplete_idx = []
        for i,d in enumerate(self.data):
            flag = False
            if isinstance(d,dict):
                if d['wikidata_id'] == None or d['caption'] == None:
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
            inputs['captions'].append(item['caption'])
            inputs['entities'].append(self.e2idx[item['wikidata_id']])
        inputs['captions'] = self.tok(text=inputs['captions'], padding=True, return_tensors='pt').to(self.dev)
        inputs['entities'] = torch.tensor(inputs['entities']).view(-1,1).to(self.dev)
        labels = torch.arange(len(batch)).to(self.dev)
        return inputs, labels
        

class LinkPredictionDataset(Dataset):

    def __init__(self, datafile: str, entity2idx: dict, rel2idx: dict = None, predict: str = 'relation'):
        assert predict in ('head', 'tail', 'relation')
        self.predict = predict
        self.e2idx = entity2idx
        self.r2idx = rel2idx
        if predict == 'relation':
            assert self.r2idx != None
        self.discarded_triples = []
        with open(datafile, 'r') as f:
            self.triples = []
            for l in f:
                discard = False
                triple = l.split()
                try:
                    h, t = self.e2idx[triple[0]], self.e2idx[triple[2]]
                    r = self.r2idx[triple[1]] if predict == 'relation' else self.e2idx[triple[1]]
                except:
                    discard = True
                    self.discarded_triples.append(l)
                if not discard:
                    self.triples.append(torch.as_tensor([h,r,t]))
        print(f'> {len(self.discarded_triples)} discarded triples due to missing mapping in the index files.')

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int):
        return self.triples[idx]

    def collate_fn(self, batch: list):
        t = torch.vstack(batch)
        if self.predict == 'relation':
            inputs, labels = t[:,[0,2]], t[:,1]
        elif self.predict == 'head':
            inputs, labels = t[:,[2,1]], t[:,0]
        elif self.predict == 'tail':
            inputs, labels = t[:,[0,1]], t[:,2]
        return inputs, labels
            

class NodeClassificationDataset(Dataset):

    def __init__(self, datafile: str, entity2idx: dict, type2idx: dict):
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
        
