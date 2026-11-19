"""Train audio or X-ray CNN from real, patient-disjoint labelled manifest data."""
import argparse,csv,hashlib,json,random,uuid
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from model import ScreeningCNN
from processing import load_audio,clean_audio,audio_tensor,load_xray,xray_tensor,SR,SECONDS,AUDIO_VERSION,XRAY_VERSION
from config import MODEL_DIR

def read_manifest(path):
    manifest=Path(path).resolve()
    with manifest.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'path','label','patient_id','split'}.issubset(reader.fieldnames or []):raise ValueError('Manifest needs path,label,patient_id,split columns.')
        rows=list(reader)
    if not rows:raise ValueError('Manifest has no real dataset rows.')
    subjects={};files=set();hashes={}
    for r in rows:
        if r['split'] not in {'train','val','test'} or not r['patient_id'].strip() or not r['label'].strip():raise ValueError('Each row needs a label, patient_id, and train/val/test split.')
        p=(manifest.parent/r['path']).resolve()
        if not p.is_file():raise ValueError(f'Dataset file not found: {p.name}')
        if p in files:raise ValueError('Duplicate file path in manifest.')
        files.add(p);r['path']=str(p)
        if r['patient_id'] in subjects and subjects[r['patient_id']]!=r['split']:raise ValueError('Patient leakage across splits. Keep every subject in a single split.')
        subjects[r['patient_id']]=r['split']
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in hashes and hashes[digest]!=r['split']:raise ValueError('Identical file content occurs across splits.')
        hashes[digest]=r['split']
    labels=sorted({r['label'] for r in rows})
    if len(labels)<2:raise ValueError('Provide at least two real classes.')
    for split in ['train','val','test']:
        if {r['label'] for r in rows if r['split']==split}!=set(labels):raise ValueError(f'Every class must occur in {split}; split by patient first.')
    return rows,labels

class Records(Dataset):
    def __init__(self,rows,labels,kind,training=False):self.rows=rows;self.labels=labels;self.kind=kind;self.training=training
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        if self.kind=='audio':
            y=clean_audio(load_audio(r['path']))
            if len(y)>SR*SECONDS:
                start=random.randrange(len(y)-SR*SECONDS+1) if self.training else 0
                y=y[start:start+SR*SECONDS]
            x=audio_tensor(y)
        else:x=xray_tensor(load_xray(r['path']))
        return x,self.labels.index(r['label'])

def evaluate(model,loader,n):
    matrix=torch.zeros((n,n),dtype=torch.int64)
    model.eval()
    with torch.inference_mode():
        for x,y in loader:
            predicted=model(x).argmax(1)
            for a,b in zip(y,predicted):matrix[int(a),int(b)]+=1
    recalls=[];f1=[]
    for i in range(n):
        tp=matrix[i,i].item();actual=matrix[i].sum().item();pred=matrix[:,i].sum().item()
        recall=tp/max(actual,1);precision=tp/max(pred,1);recalls.append(recall);f1.append(2*precision*recall/max(precision+recall,1e-12))
    return {'accuracy':matrix.diag().sum().item()/max(matrix.sum().item(),1),'balancedAccuracy':sum(recalls)/n,'macroF1':sum(f1)/n,'confusionMatrix':matrix.tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['audio','xray'],required=True);p.add_argument('--manifest',required=True);p.add_argument('--normal-label',default='Normal');p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--output',default=str(MODEL_DIR));args=p.parse_args()
    if args.epochs<1 or args.batch_size<1:p.error('epochs and batch-size must be positive')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2)
    rows,labels=read_manifest(args.manifest)
    if args.normal_label not in labels:raise ValueError('normal-label must match the dataset normal class exactly.')
    allowed={'Normal','Normal Breathing','Asthma','COPD','Pneumonia-related respiratory pattern'} if args.kind=='audio' else None
    if allowed and not set(labels).issubset(allowed):raise ValueError('Audio labels must use the quoted classes; see DATASETS.md.')
    loaders={s:DataLoader(Records([r for r in rows if r['split']==s],labels,args.kind,s=='train'),batch_size=args.batch_size,shuffle=s=='train',num_workers=0) for s in ['train','val','test']}
    model=ScreeningCNN(2 if args.kind=='audio' else 1,len(labels));optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    counts=np.array([sum(r['split']=='train' and r['label']==c for r in rows) for c in labels]);loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(len(rows)/counts,dtype=torch.float32))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);best=-1;history=[]
    for epoch in range(args.epochs):
        model.train();total=0
        for x,y in loaders['train']:
            optimizer.zero_grad();loss=loss_fn(model(x),y);loss.backward();optimizer.step();total+=float(loss.detach())
        val=evaluate(model,loaders['val'],len(labels));history.append({'epoch':epoch+1,'loss':total/len(loaders['train']),'validation':val});print(json.dumps(history[-1]),flush=True)
        if val['macroF1']>best:
            best=val['macroF1'];torch.save({'state':model.state_dict(),'labels':labels,'normal_label':args.normal_label,'preprocessing':AUDIO_VERSION if args.kind=='audio' else XRAY_VERSION,'modelVersion':str(uuid.uuid4()),'validation':val},out/f'{args.kind}.pt')
    payload=torch.load(out/f'{args.kind}.pt',map_location='cpu',weights_only=True);model.load_state_dict(payload['state']);metrics={'labels':labels,'history':history,'test':evaluate(model,loaders['test'],len(labels)),'testProtocol':'One initial 5-second segment per audio file; images use full resized view. Independent screening baseline, not clinically validated.'};(out/f'{args.kind}_metrics.json').write_text(json.dumps(metrics,indent=2));print('Saved trained model and held-out metrics to',out)
if __name__=='__main__':main()
