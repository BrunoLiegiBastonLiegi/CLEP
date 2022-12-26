import requests, time, json, os
from bs4 import BeautifulSoup
from tqdm import tqdm
from multiprocessing import Pool

os.chdir('link-prediction/')
entities = set()
relations = set()
for s in ('train', 'test', 'valid'):
    with open(s + '.txt', 'r') as f:
        for l in f:
            h,r,t = l.replace('\n','').split('\t')
            entities.add(h)
            entities.add(t)
            relations.add(r)

print(f'> {len(entities)} different entities found.')

def get_caption(entity):
    url = "https://yago-knowledge.org/resource/"
    try:
        r = requests.get(url + entity)
    except:
        return None
    if r.status_code != 200:
        print(f'> Got {r.status_code} response.')
        time.sleep(1)
        return get_caption(entity)
    soup = BeautifulSoup(r.text, 'html.parser')
    d = soup.find('div', {'class':'card-content'})
    if d is None:
        caption = None
    else:
        caption = d.p.string
        if caption is None:
            a = soup.find('div', {'class':'card-action'})
            a = a.find_all('a', href=True)
            try:
                wikidata_id = a[1]['href'][31:]
            except:
                print(a)
                print(entity)
            url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
            query = 'SELECT ?a WHERE {{ wd:{} schema:description ?a FILTER (LANG(?a) = "en").}}'.format(wikidata_id)
            r = requests.get(url, params = {'format': 'json', 'query': query})
            caption = r.json()['results']['bindings']
            #if not len(caption) > 0 :
            #    print(r.json())
            caption = caption[0]['a']['value'] if len(caption) > 0 else None
    if caption is not None:    
        caption = caption[0].upper() + caption[1:]
        if caption[-1] != '.':
            caption += '.'
    #print(caption)
    return caption

print('> Looking for the captions.')
entities = list(entities)
with Pool(12) as p:
    captions = list(tqdm(p.imap(get_caption, entities), total=len(entities)))
    
n = 0
for c in captions:
    if c is None:
        n+=1
print(f'> {n} missing captions.')

entities = {
    e: {
        'entity_id': e,
        'caption': c
    }
    for e,c in zip(entities, captions)}
with open('../entities.json', 'w') as f:
    json.dump(entities, f, indent=2)
