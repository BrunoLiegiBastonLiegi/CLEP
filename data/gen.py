import json, random, os, sys, argparse, shutil


parser = argparse.ArgumentParser(description='Data Generation')
parser.add_argument('dataset')
parser.add_argument('--cut', action='store_true')
args = parser.parse_args()

if args.dataset[-1] != '/':
    args.dataset[1] += '/'

os.chdir(args.dataset)

print(f'---- Data Generation ---- \n\n> Opening {args.dataset}entities.json .')
with open('entities.json', 'r') as f:
    ents = json.load(f)

print(f'> {len(ents)} entities found.')
print('---- Pretraining Data ----')

if args.cut:
    cut_dir = '../{}-cut'.format(args.dataset[:-1])
    os.mkdir(cut_dir)
    os.chdir(cut_dir)
    ents = {
        k: v
        for k,v in ents.items() if v['caption'] is not None
    }
    with open('entities.json', 'w') as f:
        json.dump(ents, f, indent=2)
    test = {}
else:
    test = {
        k: v
        for k,v in ents.items() if v['caption'] is None
    }
    print(f'> {len(test)} missing captions, moving them to the test set.')

[ ents.pop(k) for k in test.keys() ]

train = dict(random.sample(list(ents.items()), k=int(0.8*len(ents))))

print(f'> Generated train set of {len(train)} entities. ( {args.dataset}pretraining/train.json )')

[ ents.pop(k) for k in train.keys() ]

test.update(ents)

print(f'> Generated test set of {len(test)} entities. ( {args.dataset}pretraining/test.json )')

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

print(f'> Generated entity index file. ( {args.dataset}ent2idx.json )')
print('---- Link Prediction Data ----')
try:
    os.mkdir('link-prediction')
except:
    pass

triples = {}
relations = set()
for s in ('train', 'test', 'valid'):
    try:
        if args.cut:
            shutil.copyfile('../{}/link-prediction/{}_original.txt'.format(args.dataset,s), 'link-prediction/{}_original.txt'.format(s))
        f = open('link-prediction/{}_original.txt'.format(s), 'r')
    except:
        if args.cut:
            shutil.copyfile('../{}/link-prediction/{}.txt'.format(args.dataset,s), 'link-prediction/{}.txt'.format(s))
        os.rename('link-prediction/{}.txt'.format(s), 'link-prediction/{}_original.txt'.format(s))
        f = open('link-prediction/{}_original.txt'.format(s), 'r')
    triples[s] = []
    for l in f:
        h,r,t = l.replace('\n', '').split('\t')
        try:
            h,t = train[h]['entity_id'], train[t]['entity_id']
        except:
            continue
        triples[s].append([h, r, t])
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
    print(f'> Generated {s} set of {len(triples[s])} triples. ( {args.dataset}link-prediction/{s}.txt )')
    try:
        os.rename('link-prediction/corrupted_{}_triples+inverse.pt'.format(s), 'link-prediction/corrupted_{}_triples+inverse.pt.bak'.format(s))
        print('Found corrupted triples file, creating a backup (link-prediction/corrupted_{}_triples+inverse.pt.bak).'.format(s))
    except:
        pass
            
rel2idx = dict(zip(relations, range(len(relations))))
with open('rel2idx.json', 'w') as f:
    json.dump(rel2idx, f, indent=2)

print(f'> Generated relations index file. ( {len(rel2idx)} relations )')
    
