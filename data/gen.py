import json, random, os, sys

if sys.argv[1][-1] != '/':
    sys.argv[1] += '/'

os.chdir(sys.argv[1])

print(f'---- Data Generation ---- \n\n> Opening {sys.argv[1]}entities.json .')
with open('entities.json', 'r') as f:
    ents = json.load(f)

print(f'> {len(ents)} entities found.')
print('---- Pretraining Data ----')
test = {
    k: v
    for k,v in ents.items() if v['caption'] is None
}

print(f'> {len(test)} missing captions, moving them to the test set.')

[ ents.pop(k) for k in test.keys() ]

train = dict(random.sample(list(ents.items()), k=int(0.8*len(ents))))

print(f'> Generated train set of {len(train)} entities. ( {sys.argv[1]}pretraining/train.json )')

[ ents.pop(k) for k in train.keys() ]

test.update(ents)

print(f'> Generated test set of {len(test)} entities. ( {sys.argv[1]}pretraining/test.json )')

try:
    os.mkdir('pretraining')
except:
    pass
for s,d in zip(('train', 'test'), (train, test)):
    with open('pretraining/'+ s + '.json', 'w') as f:
        json.dump(d, f, indent=2)

train.update(test)
ent2id = dict(zip([v['entity_id'] for v in train.values()], range(len(train))))
with open ('ent2idx.json', 'w') as f:
    json.dump(ent2id, f, indent=2)

print(f'> Generated entity index file. ( {sys.argv[1]}ent2idx.json )')
print('---- Link Prediction Data ----')
try:
    os.mkdir('link-prediction')
except:
    pass

triples = {}
relations = set()
for s in ('train', 'test', 'valid'):
    try:
        f = open('link-prediction/{}_original.txt'.format(s), 'r')
    except:
        os.rename('link-prediction/{}.txt'.format(s), 'link-prediction/{}_original.txt'.format(s))
        f = open('link-prediction/{}_original.txt'.format(s), 'r')
    triples[s] = []
    for l in f:
        h,r,t = l.replace('\n', '').split('\t')
        triples[s].append([train[h]['entity_id'], r, train[t]['entity_id']])
        relations.add(r)
    f.close()
    with open('link-prediction/{}.txt'.format(s), 'w') as f:
        for l in triples[s]:
            f.write(l[0])
            f.write('\t')
            f.write(l[1])
            f.write('\t')
            f.write(l[2])
            f.write('\n')
    print(f'> Generated {s} set of {len(triples[s])} triples. ( {sys.argv[1]}link-prediction/{s}.txt )')
    try:
        os.rename('link-prediction/corrupted_{}_triples+inverse.pt'.format(s), 'link-prediction/corrupted_{}_triples+inverse.pt.bak'.format(s))
        print('Found corrupted triples file, creating a backup (link-prediction/corrupted_{}_triples+inverse.pt.bak).'.format(s))
    except:
        pass
            
rel2idx = dict(zip(relations, range(len(relations))))
with open('rel2idx.json', 'w') as f:
    json.dump(rel2idx, f, indent=2)

print(f'> Generated relations index file. ( {len(rel2idx)} relations )')
    
