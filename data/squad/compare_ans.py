import json, sys

with open('dev-v2.0.json', 'r') as f:
    data = json.load(f)['data']

id2q = {}
for article in data:
    for p in article['paragraphs']:
      for qa in p['qas']:
          plaus = []
          try:
                    plaus = qa['plausible_answers']
          except:
                    plaus = []
          id2q[qa['id']] =  {'question': qa['question'], 'plausible_answers': plaus}

with open('answers_true.json', 'r') as f:
    gt = json.load(f)

with open('answers_distilbert-base-cased_baseline.json', 'r') as f:
    base = json.load(f)

with open(sys.argv[1], 'r') as f:
    cpp = json.load(f)

    
for k in gt.keys():
    if gt[k] == "" and base[k] == "" and cpp[k] != "":
        print(id2q[k], '\n', cpp[k])
        
