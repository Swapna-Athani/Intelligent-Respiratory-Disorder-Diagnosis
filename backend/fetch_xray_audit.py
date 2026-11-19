"""Acquire a fresh, image-disjoint audit set; do not use it for model selection."""
import collections,csv,hashlib,json,random,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fetch_datasets import fetch
from processing import load_xray

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/public_datasets'

def main():
    with (DATA/'inventory.csv').open(newline='',encoding='utf-8') as f:existing=list(csv.DictReader(f))
    seen={r['sha256'] for r in existing};names={Path(r['path']).name.lower() for r in existing if r['modality']=='xray'}
    rng=random.Random(8301);selected=[];targets={};accepted=collections.Counter();excluded=[]
    for source,label,count in [('pneumonia','Normal',10),('tb','Normal',10),('pneumonia','Pneumonia',20),('tb','Tuberculosis',20)]:
        index=json.loads((DATA/'metadata'/(source+'_index.json')).read_text())
        unique={}
        for item in index:
            p=Path(item['name'])
            if '__MACOSX' in item['name'] or p.name.startswith('.') or p.parent.name.lower()!=label.lower() or p.suffix.lower() not in {'.png','.jpg','.jpeg'} or p.name.lower() in names:continue
            unique.setdefault(p.name.lower(),item['name'])
        items=sorted(unique.values());rng.shuffle(items)
        targets[(source,label)]=count
        for name in items[:count+10]:selected.append({'source':source,'label':label,'source_file':name})
    target=DATA/'fresh_xray_audit';target.mkdir(exist_ok=True)
    def download(row):
        path=target/row['source']/row['label']/Path(row['source_file']).name
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(fetch(row['source'],row['source_file']))
        load_xray(path)
        row.update(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(download,selected):
            key=(row['source'],row['label'])
            if accepted[key]>=targets[key]:continue
            if row['sha256'] in seen:
                excluded.append({'source_file':row['source_file'],'reason':'exact duplicate; excluded before evaluation'})
                continue
            seen.add(row['sha256']);rows.append(row);accepted[key]+=1
            print(f'Audit accepted {len(rows)}/60',flush=True)
    assert len(rows)==60 and dict(accepted)==targets,'Insufficient unique audit images'
    (target/'excluded_duplicates.json').write_text(json.dumps(excluded,indent=2),encoding='utf-8')
    with (target/'manifest.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['source','label','source_file','path','sha256']);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'fresh_images':len(rows),'counts':dict(collections.Counter(r['label'] for r in rows)),'patient_independence':False}),flush=True)

if __name__=='__main__':main()
