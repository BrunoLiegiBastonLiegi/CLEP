import torch, random

candidates = torch.arange(20)
relations = torch.arange(4)
filter = {(random.choice(candidates).item(), random.choice(relations).item(), random.choice(candidates).item()) for i in range(1000)}
filter = torch.as_tensor([list(i) for i in filter])
print('Number of filter triples: ', filter.shape[0])

triples = filter[torch.randperm(filter.shape[0])][:10]

mode = 'tail'
idx, idx_pair = (0, [0,1]) if mode == 'tail' else (2, [1,2])
mask = (triples[:,idx].view(-1,1) == candidates) 
scores = torch.randn(10, candidates.shape[0])

# unfiltered
ranks = (scores.sort(dim=-1, descending=True).indices == mask.nonzero()[:,1].view(-1,1)).nonzero()[:,1]
print(ranks)

# filtered
filter_mask = (triples.view(-1,1,3)[:,:,idx_pair].detach().cpu() == filter[:,idx_pair]).all(-1)
tmp_cand = candidates.detach().cpu()

idx = 0 if mode == 'head' else 2
#for i in range(filter_mask.shape[0]):
#    print(filter[filter_mask[i]][:,idx])
#    print((filter[filter_mask[i]][:,idx].view(-1,1) == tmp_cand).sum(0).bool().nonzero())
filter_mask = torch.vstack([
    (filter[filter_mask[i]][:,idx].view(-1,1) == tmp_cand).sum(0).bool()
    for i in range(filter_mask.shape[0])
])
filter_mask = (mask.logical_not() * filter_mask.to(mask.device)).bool()
del tmp_cand
scores[filter_mask] = -1e8
#print(scores.sort(dim=-1, descending=True))
ranks = (scores.sort(dim=-1, descending=True).indices == mask.nonzero()[:,1].view(-1,1)).nonzero()[:,1]
print(ranks)
