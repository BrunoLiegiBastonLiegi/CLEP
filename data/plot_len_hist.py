import json
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side, tokenizer.pad_token = 'left', tokenizer.bos_token

with open('FB15k-237/pretraining/entities.json', 'r') as f:
    fbcap = json.load(f)
    fbcap = [v['caption'] for v in fbcap.values() if v is not None]
    fbcaplen = [len(tokenizer(c)['input_ids']) for c in fbcap if c is not None]
with open('WN18RR/pretraining/entities.json', 'r') as f:
    wncap = json.load(f)
    wncap = [v['caption'] for v in wncap.values() if v is not None]
    wncaplen = [len(tokenizer(c)['input_ids']) for c in wncap if c is not None]
    
def plot_hist(d1, d2, bins=50):
    plt.hist(d1, bins=bins, alpha=0.5, density=True)
    plt.hist(d2, bins=2*bins, alpha=0.5, density=True)
    plt.show()

plot_hist(fbcaplen, wncaplen, bins=30)

cut = 15
wncaplen_cut = [l for l in wncaplen if l <= cut]
fbcaplen_cut = [l for l in fbcaplen if l <= cut]
print(f'Number of captions with len < {cut} in WN18RR:\t{len(wncaplen_cut)} ({int(100*len(wncaplen_cut)/len(wncaplen))}%)')
print(f'Number of captions with len < {cut} in FB15k-237:\t{len(fbcaplen_cut)} ({int(100*len(fbcaplen_cut)/len(fbcaplen))}%)')
 
plot_hist(fbcaplen_cut, wncaplen_cut, bins=5)
