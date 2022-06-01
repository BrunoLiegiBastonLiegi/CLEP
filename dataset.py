from torch.utils.data import Dataset
import pickle, torch


class CLIPDataset(Dataset):

    def __init__(self, datafile: str, tokenizer, entity2idx: dict, device: torch.device = torch.device('cpu')):
        with open(datafile, 'rb') as f:
            self.data = list(pickle.load(f).values())
            self.discard_incomplete()
        self.tok = tokenizer
        self.e2idx = entity2idx
        self.dev = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def discard_incomplete(self):
        incomplete_idx = []
        for i,d in enumerate(self.data):
            flag = False
            if isinstance(d,dict):
                if 'embedding' in d.keys():
                    if isinstance(d['embedding'],type(None)) or d['wikidata_id'] == None or d['caption'] == None:
                        flag = True
                else:
                    flag = True    
            else:
                flag = True
            incomplete_idx.append(flag)
        self.data = [ d for d,f in zip(self.data, incomplete_idx) if not f ]

    def collate_fn(self, batch: list):
        inputs = {'captions':[], 'entities':[]}
        for item in batch:
            inputs['captions'].append(item['caption'])
            inputs['entities'].append(self.e2idx[item['wikidata_id']])
        inputs['captions'] = self.tok(text=inputs['captions'], padding=True, return_tensors='pt').to(self.dev)
        inputs['entities'] = torch.tensor(inputs['entities']).view(-1,1).to(self.dev)
        labels = torch.arange(len(batch)).to(self.dev)
        return inputs, labels
        

