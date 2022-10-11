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
            for k,v in epoch[t].items():
                dest[t][k].append(v)

for m in metrics:
    append_metrics(m['RGCN Baseline'], baseline)
    append_metrics(m['RGCN with Caption Pretraining'], caption_pretraining)

#print(json.dumps(baseline,indent=2))

def reshape(d, l):
    for t in ('raw', 'filtered'):
        for k,v in d[t].items():
            d[t][k] = torch.as_tensor(v).view(l,-1) 

reshape(baseline, len(metrics))
reshape(caption_pretraining, len(metrics))

#print(baseline)

metric_type = 'filtered'

fig, axs = plt.subplots(2, 3, figsize=(16,9))

for metric, ax in zip(baseline[metric_type].keys(), axs.flatten()[:-1]):
    for d in (baseline, caption_pretraining):
        mean = d[metric_type][metric].mean(0)
        #top = d[metric_type][metric].max(0).values
        #bottom = d[metric_type][metric].min(0).values
        std = d[metric_type][metric].std(0)
        ax.plot(mean)
        ax.fill_between(range(len(metrics)), mean+std, mean-std, alpha=0.3)
        #ax.plot(caption_pretraining[metric_type][metric].mean(0), c='orange')
        ax.set_title(metric)
plt.savefig('lp_metrics.pdf', dpi=300, format='pdf', bbox_inches='tight')
plt.show()
