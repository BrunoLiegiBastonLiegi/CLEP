from nltk.corpus import wordnet as wn
import sys, json

entities = {}
n = 0
for in_f in sys.argv[1:]:
    with open(in_f, 'r') as f:
        for line in f:
            h,r,t = line[:-1].split('\t')
            print(h, r, t)
            for e in (h,t):
                if e not in entities.keys():
                    cap = wn.synset(e).definition()
                    cap = cap[0].upper() + cap[1:]
                    if cap[-1] != '.':
                        cap += '.'
                    entities[e] = {
                        'entity_id': e,
                        'index': n,
                        'caption': cap
                    }
                    n += 1

with open('entities.json', 'w') as f:
    json.dump(entities, f, indent=2)
