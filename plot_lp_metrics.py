import json
import matplotlib.pyplot as plt

with open('lp_metrics.json', 'r') as f:
    metrics = json.load(f)

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

append_metrics(metrics['RGCN Baseline'], baseline)
append_metrics(metrics['RGCN with Caption Pretraining'], caption_pretraining)


metric_type = 'filtered'

fig, axs = plt.subplots(2, 3, figsize=(16,9))

for metric, ax in zip(baseline[metric_type].keys(), axs.flatten()[:-1]):
    ax.plot(baseline[metric_type][metric], c='blue')
    ax.plot(caption_pretraining[metric_type][metric], c='orange')
    ax.set_title(metric)
plt.savefig('lp_metrics.pdf', dpi=300, format='pdf', bbox_inches='tight')
plt.show()
