import json

baseline, caption = [], []
for i in range(3):
	with open('results_Baseline_{}.json'.format(i), 'r') as f:
		baseline.append(json.load(f))
	with open('results_Caption_Pretraining_{}.json'.format(i), 'r') as f:
		caption.append(json.load(f))

results = []
for bas, cap in zip(baseline, caption):
	d = {}
	for name,res in zip(('CompGCN_baseline', 'CompGCN_caption_pretraining'), (bas['validation'], cap['validation'])):
		d[name] = {}
		for k,v in res.items():
			d[name][k] = {'mrr':v['mrr'], 'mean_rank':v['mr'], 'hits@1':v['hits@1'], 'hits@3':v['hits@3'], 'hits@10':v['hits@10']}
	results.append(d)

for i in range(len(results)):
	with open('lp_results_CompGCN_{}.json'.format(i), 'w') as f:
		json.dump(results[i], f, indent=2)
