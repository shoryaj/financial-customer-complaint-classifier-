"""Download an API result window, not a random full-database sample."""
import argparse,json,urllib.request,datetime
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=10000);a=p.parse_args()
if not 1<=a.size<=10000:raise ValueError('Use 1–10000 rows.')
u=f'https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?size={a.size}&no_aggs=true&has_narrative=true'
r=json.load(urllib.request.urlopen(u,timeout=120))
d=pd.DataFrame([x['_source'] for x in r['hits']['hits']]);d=d.rename(columns={'complaint_what_happened':'Consumer complaint narrative','product':'Product','complaint_id':'Complaint ID','date_received':'Date received'})
Path('data').mkdir(exist_ok=True);d[['Consumer complaint narrative','Product','Complaint ID','Date received']].to_csv('data/complaints.csv',index=False)
Path('data/provenance.json').write_text(json.dumps({'url':u,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':len(d),'selection':'First API result window under server default ordering; not a random database sample.'},indent=2))
print('Downloaded',len(d),'rows')
