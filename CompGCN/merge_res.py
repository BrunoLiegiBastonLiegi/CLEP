import json, sys

with open(sys.argv[1], 'r') as f:
    pret = json.load(f)
with open(sys.argv[2], 'r') as f:
    base = json.load(f)

d = {'CompGCN Caption Pretraining': pret, 'CompGCN Baseline': base}

with open(sys.argv[3], 'w') as f:
    json.dump(d, f, indent=2)
