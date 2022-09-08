import torch

nodes = torch.tensor(range(5))
check_triples = ((5*torch.randn(10,3))**2).sqrt().long()
check_triples[check_triples > 4] = 4
check_triples = torch.unique(check_triples, dim=0)
true_batch = check_triples[[0,4,5,8],:].clone()

heads = torch.vstack([nodes for j in range(true_batch.shape[0])]).T.reshape(-1,1)
batch = torch.vstack([true_batch for j in range(nodes.shape[0])])
batch[:,0] = heads[:,0]

#print(batch.shape, check_triples.shape, int(batch.shape[0]/check_triples.shape[0]))

#ratio = int(batch.shape[0]/check_triples.shape[0])

#tmp = torch.vstack([check_triples for i in range(ratio)])
#print(batch == tmp)

check_triples = {tuple(t.tolist()) for t in check_triples}
batch = {tuple(t.tolist()) for t in batch}
keep = torch.as_tensor(list((batch.symmetric_difference(check_triples)).intersection(batch)))
batch = torch.as_tensor(list(batch))
check_triples = torch.as_tensor(list(check_triples))
print(f'> Check_triples:\n {check_triples}')
print(f'> True Batch:\n {true_batch}')
print(f'> Corrupted triples:\n {keep}')
scores = torch.sigmoid(torch.randn(keep.shape[0]+true_batch.shape[0]))
in_triples = torch.vstack((true_batch, keep))
size = true_batch.shape[0]

f = lambda x : (scores[ # take only the scores of the relevant triples, i.e. those with the same head-rel or rel-tail
    torch.cat(( # concatenate the true triple (pos 0) and all the corrupted ones
        torch.nn.functional.one_hot(torch.tensor(x[0]), num_classes=size).bool(), # true triple mask
        (in_triples[size:,1:3] == x[1][1:3]).all(-1) # relevant corrupted triples mask
    ), dim=0)
].sort(descending=True).indices == 0).nonzero() # sort the scores and look for the position of the true triple

ranks = (torch.vstack(list(map(
    f,
    enumerate(in_triples[:size])
))).flatten())

trip_scores = torch.hstack((in_triples, scores.view(-1,1))) 
print(f'> Batch:\n {trip_scores}')
print(f'> Ranks:')
for i, (t, r) in enumerate(zip(in_triples[:size], ranks)):
    print(trip_scores[torch.cat((torch.nn.functional.one_hot(torch.tensor(i), num_classes=size).bool(), (in_triples[size:,1:3] == t[1:3]).all(-1)))])
    print(t, r, '\n')

#for t in keep:
#    print(t.tolist())
#    print('Position in batch:', (t == batch).all(1).nonzero().tolist())
#    print('position in check_triples: ', (t == check_triples).all(1).nonzero().tolist())
