import json, sys, torch
import matplotlib.pyplot as plt

metrics = []
for filename in sys.argv[1:]:
    with open(filename, 'r') as f:
        metrics.append(json.load(f))
        
baseline = {
    'raw': {
        'mean_rank': [],
        'mrr': [],
        'hits@1': [],
        'hits@3': [],
        'hits@10': [],
    },
    'filtered': {
        'mean_rank': [],
        'mrr': [],
        'hits@1': [],
        'hits@3': [],
        'hits@10': [],
    }
}
caption_pretraining = {
    'raw': {
        'mean_rank': [],
        'mrr': [],
        'hits@1': [],
        'hits@3': [],
        'hits@10': [],
    },
    'filtered': {
        'mean_rank': [],
        'mrr': [],
        'hits@1': [],
        'hits@3': [],
        'hits@10': [],
    }
}

def append_metrics(source, dest):
    for epoch in source.values():
        for t in ('raw', 'filtered'):
            try:
                items = epoch[t].items()
            except:
                try:
                    items = epoch.items()
                except:
                    continue
            for k,v in items:
                dest[t][k].append(v)
                
max_n_epochs = 0
for m in metrics:
    for i in (0,1):
        if 'filtered' in list(m.values())[i].keys():
            max_n_epochs = max(len(list(m.values())[i]['filtered'].keys()), max_n_epochs)
        else:
            max_n_epochs = max(len(list(m.values())[i].keys()), max_n_epochs)
            
for m in metrics:
    for i, d in zip((0, 1), (baseline, caption_pretraining)):
        length = len(list(m.values())[i]['filtered'].keys()) if 'filtered' in list(m.values())[i].keys() else len(list(m.values())[i].keys())
        if length < max_n_epochs:
            for n in range(length, max_n_epochs):
                try:
                    last = list(list(m.values())[i]['filtered'].values())[-1]
                    list(m.values())[i]['filtered'][n] = last
                except:
                    last = list(list(m.values())[i].values())[-1]
                    list(m.values())[i][n] = last
        append_metrics(list(m.values())[i], d)
    #append_metrics(list(m.values())[1], caption_pretraining)

#print(json.dumps(baseline,indent=2))
    
def reshape(d, l):
    for t in ('raw', 'filtered'):
        for k,v in d[t].items():
            d[t][k] = torch.as_tensor(v).view(l,-1)

reshape(baseline, len(metrics))
reshape(caption_pretraining, len(metrics))

metric_type = 'filtered'

fig, axs = plt.subplots(2, 3, figsize=(16,9))

for metric, ax in zip(baseline[metric_type].keys(), axs.flatten()[:-1]):
    for d in (baseline, caption_pretraining):
        mean = d[metric_type][metric].mean(0)
        #top = d[metric_type][metric].max(0).values
        #bottom = d[metric_type][metric].min(0).values
        std = d[metric_type][metric].std(0)
        ax.plot(mean)
        ax.fill_between(range(max_n_epochs), mean+std, mean-std, alpha=0.3)
        #ax.fill_between(range(len(metrics)), mean+std, mean-std, alpha=0.3)
        #ax.plot(caption_pretraining[metric_type][metric].mean(0), c='orange')
        ax.set_title(metric)
plt.savefig('lp_metrics.pdf', dpi=300, format='pdf', bbox_inches='tight')
plt.show()
