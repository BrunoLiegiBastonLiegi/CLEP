import pickle, sys, requests, tqdm, time
from multiprocessing import Pool, Manager
from lxml.html import fromstring
from bs4 import BeautifulSoup

def make_request(url: str, query: str, **kwargs):
    agents = [
        {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8", 
         "Accept-Encoding": "gzip, deflate, br", 
         "Accept-Language": "it-IT,it;q=0.8,en-US;q=0.5,en;q=0.3", 
         "Host": "httpbin.org", 
         "Referer": "https://www.scrapehero.com/", 
         "Sec-Fetch-Dest": "document", 
         "Sec-Fetch-Mode": "navigate", 
         "Sec-Fetch-Site": "cross-site", 
         "Sec-Fetch-User": "?1", 
         "Upgrade-Insecure-Requests": "1", 
         "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:99.0) Gecko/20100101 Firefox/99.0"},
        {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9", 
         "Accept-Encoding": "gzip, deflate, br", 
         "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7", 
         "Host": "httpbin.org", 
         "Sec-Ch-Ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"101\", \"Google Chrome\";v=\"101\"", 
         "Sec-Ch-Ua-Mobile": "?0", 
         "Sec-Ch-Ua-Platform": "\"Linux\"", 
         "Sec-Fetch-Dest": "document", 
         "Sec-Fetch-Mode": "navigate", 
         "Sec-Fetch-Site": "none", 
         "Sec-Fetch-User": "?1", 
         "Upgrade-Insecure-Requests": "1", 
         "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36"}
    ]
    try:
        proxies = kwargs['proxies']
    except:
        proxies = get_proxies()
    assert len(proxies) > 0
    proxy = random.choice(list(proxies))
    try:
        r = requests.get(
            url,
            headers = random.choice(agents),
            proxies = {"http": proxy, "https": proxy},
            params = {'format': 'json', 'query': query}
        )
        print(r.url)
    except:
        proxies.remove(proxy)
        return make_request(url, query, proxies = proxies)
    return r

def collect_caption(wid: list) -> dict :
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'#'https://query.wikidata.org/sparql'
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}
    query = 'SELECT * WHERE {' + ' UNION '.join(['{{ wd:{} schema:description ?{} }}'.format(e,e) for e in wid]) + 'FILTER ( ' + ' || '.join(['lang(?{}) = "en"'.format(e) for e in wid]) + ' )}'
    t = time.time()
    r = requests.get(url, headers=headers, params = {'format': 'json', 'query': query})
    #print(f'> Request time: {time.time()-t:.2f}')
    if r.status_code != 200: #== 429:
        try:
            timeout = int(r.headers['retry-after'])
        except:
            timeout = 60
        print(f'> Got {r.status_code} response, sleeping for {timeout} sec.')
        time.sleep(timeout)
        return collect_caption(wid)
    assert r.status_code == 200, f"Received {r.status_code} response code."
    res = dict(zip(wid, [None for i in range(len(wid))]))
    for b in r.json()['results']['bindings']:
        cap = list(b.values())[0]['value']
        res[list(b.keys())[0]] = cap[0].upper() + cap[1:] + '.'
    return res

def init_process(args):
    global entities
    entities = args
    
def main(infile: str, nproc: int=3) -> None:
    global entities
    print(f'Reading file \'{infile}\'.')
    #with open(infile, 'rb') as f:
    #    nodes = list(pickle.load(f).keys())
    with open(infile, 'r') as f:
        nodes = f.read().split('\n')
    entities.update(zip(nodes, range(len(nodes))))
    del nodes

    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    batchsize = 64
    wid_batches = list(batch(list(entities.keys()), batchsize))
    with Pool(nproc, initializer=init_process, initargs=(entities,)) as p:
        wd2cap = list(tqdm.tqdm(
            p.imap_unordered(
                collect_caption,
                wid_batches
            ),
            total=len(wid_batches)
        ))
        #for _ in tqdm.tqdm(p.imap_unordered(collect_caption, wid_batches), total=len(wid_batches)):
        #    pass

    wd2cap = {k:v for d in wd2cap for k,v in d.items()}

    for k in entities.keys():
        if k != None:
            entities[k] = {
                'wikidata_id': k,
                'caption': wd2cap[k]
            }

if __name__ == '__main__':
    
    infile = sys.argv[1]
    ofile = sys.argv[2]

    manager = Manager()
    entities = manager.dict() # Global var for storing captions and mappings found already
    
    main(infile, nproc=12)

    with open(ofile, 'w') as f:
        json.dump(entities.copy(), f, indent=4)
