import json, sys, torch, argparse
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})
import numpy as np

parser = argparse.ArgumentParser(description='Plotting results.')
parser.add_argument('in_files', nargs='+')
parser.add_argument('--log_scale', action='store_true')

args = parser.parse_args()

metrics = []
for filename in args.in_files:
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

def append_metrics(source, dest, keys=('mean_rank', 'mrr', 'hits@1', 'hits@3', 'hits@10')):
    for epoch in source.values():
        for t in ('raw', 'filtered'):
            try:
                items = epoch[t]#.items()
            except:
                try:
                    items = epoch#.items()
                except:
                    continue
            #for k,v in items:
            
            for k in keys:
                if k == 'mean_rank':
                    try:
                        dest[t][k].append(items[k])
                    except:
                        dest[t][k].append(items['mr'])
                else:
                    dest[t][k].append(items[k])

test = []
for m in metrics:
    d = {}
    for k,v in m.items():
        if 'test' in v.keys():
            d.update({k:v.pop('test')})
        else:
            ep = input(f'Missing \'test\' key, specify which epoch to calculate performance at.\n Possible choiches: {list(v.keys())}\n(Default: last one)\n> ')
            if ep == '':
                ep = list(v.keys())[-1]
            d.update({k:v[ep]})
    test.append(d)
max_n_epochs = 0
for m in metrics:
    for i in (0,1):
        if 'filtered' in list(m.values())[i].keys():
            max_n_epochs = max(len(list(m.values())[i]['filtered'].keys()), max_n_epochs)
        else:
            max_n_epochs = max(len(list(m.values())[i].keys()), max_n_epochs)
            
for m in metrics:
    idx = {}
    for i,k in enumerate(m.keys()):
        if 'baseline' in k.lower():
            idx['baseline'] = i
        elif 'pretraining' in k.lower():
            idx['pretrained'] = i
    #for i, d in zip((0, 1), (baseline, caption_pretraining)):
    for i, d in zip((idx['baseline'], idx['pretrained']), (baseline, caption_pretraining)):
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

fig, axs = plt.subplots(1, 4, figsize=(38,9))
if args.log_scale:
    scale = lambda x: np.log(x + 1e-6)
else:
    scale = lambda x: x
    
for metric, ax in zip([ k for k in baseline[metric_type].keys() if k!='hits@3'] , axs.flatten()):
    for d in (baseline, caption_pretraining):
        mean = d[metric_type][metric].mean(0)
        #top = d[metric_type][metric].max(0).values
        #bottom = d[metric_type][metric].min(0).values
        std = d[metric_type][metric].std(0)
        ax.plot(scale(mean))
        ax.fill_between(range(max_n_epochs), scale(mean+std), scale(mean-std), alpha=0.3)
        #ax.fill_between(range(len(metrics)), mean+std, mean-std, alpha=0.3)
        #ax.plot(caption_pretraining[metric_type][metric].mean(0), c='orange')
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
plt.savefig('lp_metrics.pdf', dpi=300, format='pdf', bbox_inches='tight')
plt.show()


# Ranks histogram and test metrics

test_baseline, test_cpp = {}, {}
for k in ('ranks','mean_rank', 'mrr', 'hits@1', 'hits@3', 'hits@10'):
    test_baseline.update({k:[]})
    test_cpp.update({k:[]})
    for t in test:
        for model_name, v in t.items():
            try:
                metrics = v['filtered']
            except:
                metrics = v
            if k == 'ranks':
                try:
                    d = {k: metrics['right_ranks'] + metrics['left_ranks']}
                except:
                    print('No data for ranks found.')
                    continue
            elif k == 'mean_rank':
                try:
                    d = {k:metrics[k]}
                except:
                    d = {k:metrics['mr']}
            else:
                 d = {k:metrics[k]}   
                
            if 'baseline' in model_name.lower():
                test_baseline[k].append(d[k])
            else:
                test_cpp[k].append(d[k])
                

print('\t\t Baseline\t Caption Pretrained')

for k in ('mean_rank', 'mrr', 'hits@1', 'hits@3', 'hits@10'):
    if k == 'mean_rank':
        print(f'{k}\t {int(np.mean(test_baseline[k]))}\t\t {int(np.mean(test_cpp[k]))}')
    else:
        print(f'{k}\t\t {np.mean(test_baseline[k]):.3f}\t\t {np.mean(test_cpp[k]):.3f}')

if len(test_baseline['ranks']) > 0:
    plt.rcParams.update({'font.size': 42})
    fig, ax = plt.subplots(1, 1, figsize=(20,16))
    bins = 50#'auto'
    cut = 25
    dens = True
    ranks_baseline = torch.as_tensor(test_baseline['ranks']).flatten()
    #ranks_baseline = ranks_baseline[ranks_baseline < cut]
    ranks_cpp = torch.as_tensor(test_cpp['ranks']).flatten()
    #ranks_cpp = ranks_cpp[ranks_cpp < cut]
    ax.hist(ranks_baseline.numpy(), density=dens, alpha=0.3, bins=bins)
    ax.hist(ranks_cpp.numpy(), density=dens, alpha=0.3, bins=bins)
    #plt.xlim(0,cut)
    #plt.xscale('log')
    plt.xlabel('Ranks')
    plt.ylabel('P')
    plt.yscale('log')
    plt.savefig('ranks_hist.pdf', format='pdf', dpi=300)
    plt.show()
