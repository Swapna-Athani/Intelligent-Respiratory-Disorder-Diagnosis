"""Train an exploratory X-ray model from labelled public files.
This separate prototype workflow uses image-level splits because verified patient
identifiers are absent. It does NOT provide patient-independent validation.
The strict patient-disjoint train.py workflow is unchanged.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor
from model import ScreeningCNN
from train import Records, evaluate
from processing import XRAY_VERSION
from config import MODEL_DIR

ROOT=Path(__file__).resolve().parent
WARNING='Exploratory prototype trained on a small public image subset. Evaluation uses image-level splits; patient overlap and source bias cannot be excluded. Not clinically validated. Do not use this result for medical decisions.'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',default=str(ROOT/'data/public_datasets/inventory.csv'))
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1:parser.error('epochs and batch size must be positive')
    torch.set_num_threads(2);torch.manual_seed(5785);np.random.seed(5785);rng=random.Random(5785)
    with Path(args.inventory).open(newline='',encoding='utf-8-sig') as handle:
        source=list(csv.DictReader(handle))
    unique={}; duplicates=0
    for row in source:
        if row['modality']!='xray':continue
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to((ROOT/'data').resolve()):raise ValueError('File outside project data folder')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row['sha256']:raise ValueError('File checksum mismatch')
        if digest in unique:
            if unique[digest]['label']!=row['label']:raise ValueError('Conflicting labels on identical images')
            duplicates+=1;continue
        unique[digest]={'path':str(path),'label':row['label'],'source':row['source'],'sha256':digest}
    strata=collections.defaultdict(list)
    for row in unique.values():strata[(row['source'],row['label'])].append(row)
    rows=[]
    for key,group in sorted(strata.items()):
        group.sort(key=lambda row:row['sha256']);rng.shuffle(group)
        if len(group)<10:raise ValueError('At least 10 distinct images are required per source/class')
        n=len(group);nval=max(1,round(n*.15));ntest=max(1,round(n*.15));ntrain=n-nval-ntest
        for index,row in enumerate(group):
            row['split']='train' if index<ntrain else 'val' if index<ntrain+nval else 'test'
            rows.append(row)
    labels=sorted({row['label'] for row in rows})
    if labels!=['Normal','Pneumonia','Tuberculosis']:raise ValueError('Requires real Normal, Pneumonia and Tuberculosis images')
    for split in ('train','val','test'):
        if {row['label'] for row in rows if row['split']==split}!=set(labels):raise ValueError('Each split must cover all labels')
    out=MODEL_DIR/'xray_prototype_run';out.mkdir(parents=True,exist_ok=True)
    with (out/'image_split_manifest.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['path','label','source','sha256','split']);writer.writeheader();writer.writerows(rows)
    print(WARNING,flush=True)
    counts={split:dict(collections.Counter(row['label'] for row in rows if row['split']==split)) for split in ('train','val','test')}
    print(json.dumps({'split_unit':'image','patient_independent':False,'counts':counts,'exact_duplicates_removed':duplicates}),flush=True)
    cache=out/'preprocessed';cache.mkdir(exist_ok=True)
    version=hashlib.sha256(XRAY_VERSION.encode()).hexdigest()[:12]
    def prepare(row):
        path=cache/(row['sha256']+'_'+version+'.pt')
        if path.exists():tensor=torch.load(path,map_location='cpu',weights_only=True)
        else:
            tensor=Records([row],labels,'xray')[0][0]
            torch.save(tensor,path)
        return row['sha256'],tensor
    tensors={}
    print('Preprocessing images once; cached tensors are reused for all epochs.',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index,(digest,tensor) in enumerate(pool.map(prepare,rows),1):
            tensors[digest]=tensor
            if index%25==0 or index==len(rows):print(f'Prepared {index}/{len(rows)} images',flush=True)
    loaders={}
    for split in ('train','val','test'):
        records=[row for row in rows if row['split']==split]
        dataset=TensorDataset(torch.stack([tensors[row['sha256']] for row in records]),torch.tensor([labels.index(row['label']) for row in records]))
        loaders[split]=DataLoader(dataset,batch_size=args.batch_size,shuffle=split=='train',num_workers=0)

    model=ScreeningCNN(1,len(labels));optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    class_counts=np.array([sum(row['split']=='train' and row['label']==label for row in rows) for label in labels])
    loss_fn=torch.nn.CrossEntropyLoss(weight=torch.tensor(1/class_counts,dtype=torch.float32))
    best=-1;history=[];candidate=out/'best.pt'
    for epoch in range(args.epochs):
        model.train();total=0
        for x,y in loaders['train']:
            optimizer.zero_grad();loss=loss_fn(model(x),y);loss.backward();optimizer.step();total+=float(loss.detach())
        val=evaluate(model,loaders['val'],len(labels))
        step={'epoch':epoch+1,'loss':total/len(loaders['train']),'validation':val};history.append(step);print(json.dumps(step),flush=True)
        if val['macroF1']>best:
            best=val['macroF1']
            torch.save({'state':model.state_dict(),'labels':labels,'normal_label':'Normal','preprocessing':XRAY_VERSION,'modelVersion':str(uuid.uuid4()),'validation':val,'researchWarning':WARNING,'evaluationScope':'image-level; patient independence not established'},candidate)
    payload=torch.load(candidate,map_location='cpu',weights_only=True);model.load_state_dict(payload['state'])
    test=evaluate(model,loaders['test'],len(labels))
    metrics={'labels':labels,'createdAt':datetime.now(timezone.utc).isoformat(),'splitUnit':'image','patientIndependent':False,'researchWarning':WARNING,'counts':counts,'history':history,'test':test,'testProtocol':'Exact-byte-disjoint image splits, stratified by source and class. Patient overlap and near-duplicates are not ruled out.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    payload['test']=test
    tb_index=labels.index('Tuberculosis');tb_cases=sum(test['confusionMatrix'][tb_index]);tb_recall=test['confusionMatrix'][tb_index][tb_index]/max(tb_cases,1)
    payload['researchWarning']=WARNING+f" Image-level test accuracy: {test['accuracy']*100:.1f}% on {len(loaders['test'].dataset)} images; TB recall: {tb_recall*100:.1f}% on {tb_cases} images."
    # Install only a complete checkpoint. The live API loads it on the next request.
    temporary=MODEL_DIR/'xray.pt.pending';torch.save(payload,temporary);os.replace(temporary,MODEL_DIR/'xray.pt')
    (MODEL_DIR/'xray_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps({'installed':str(MODEL_DIR/'xray.pt'),'test':test,'warning':WARNING}),flush=True)


if __name__=='__main__':main()
