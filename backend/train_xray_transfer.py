"""Train a frozen ImageNet ResNet18 feature extractor with a supervised X-ray head.
Keeps the original image split; uses validation only for checkpoint selection.
A fresh audit is evaluated only after selection. Patient independence is unknown.
"""
import collections,csv,hashlib,json,os,random,shutil,uuid
from datetime import datetime,timezone
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from model import ScreeningCNN
from processing import load_xray
from train import evaluate
from config import MODEL_DIR
from xray_transfer import XrayTransferModel,transfer_tensor,ARCHITECTURE,PREPROCESSING

ROOT=Path(__file__).resolve().parent
WARNING='Exploratory public-data model. Evaluation is image-level; patient overlap, near-duplicates and source/age bias cannot be excluded. Not clinically validated. Do not use for medical decisions.'

def csv_rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as handle:return list(csv.DictReader(handle))

def main():
    torch.set_num_threads(min(4,os.cpu_count() or 2));torch.manual_seed(8301);random.seed(8301)
    rows=csv_rows(MODEL_DIR/'xray_prototype_run/image_split_manifest.csv')
    labels=sorted({row['label'] for row in rows});assert labels==['Normal','Pneumonia','Tuberculosis']
    baseline=torch.load(MODEL_DIR/'xray.pt',map_location='cpu',weights_only=True)
    seen={};groups=collections.defaultdict(list)
    for row in rows:
        path=Path(row['path']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row['sha256'] or digest in seen:raise ValueError('Invalid checksum or duplicate image in fixed split')
        seen[digest]=row['split'];groups[row['split']].append(row)
    out=MODEL_DIR/'xray_transfer_run';out.mkdir(parents=True,exist_ok=True)
    cache=out/'features';cache.mkdir(exist_ok=True)
    model=XrayTransferModel(len(labels),pretrained=True).eval()
    def extract(records,title):
        vectors=[]
        for start in range(0,len(records),8):
            batch=records[start:start+8];missing=[];positions=[];parts=[None]*len(batch)
            for i,row in enumerate(batch):
                path=cache/(row['sha256']+'.pt')
                if path.exists():parts[i]=torch.load(path,map_location='cpu',weights_only=True)
                else:missing.append(transfer_tensor(load_xray(row['path'])));positions.append(i)
            if missing:
                with torch.no_grad():features=model.backbone(torch.stack(missing))
                for position,vector in zip(positions,features):
                    parts[position]=vector.clone();torch.save(parts[position],cache/(batch[position]['sha256']+'.pt'))
            vectors.extend(parts)
            if start%40==0 or start+8>=len(records):print(f'{title}: features {min(start+8,len(records))}/{len(records)}',flush=True)
        return torch.stack(vectors)
    features=extract(rows,'Fixed split')
    train_indices=[i for i,row in enumerate(rows) if row['split']=='train']
    mean=features[train_indices].mean(0);scale=features[train_indices].std(0).clamp_min(.1)
    normalized=(features-mean)/scale
    loaders={}
    for split in ('train','val','test'):
        indices=[i for i,row in enumerate(rows) if row['split']==split]
        loaders[split]=DataLoader(TensorDataset(normalized[indices],torch.tensor([labels.index(rows[i]['label']) for i in indices])),batch_size=32,shuffle=split=='train')
    head=nn.Linear(512,len(labels));optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.1)
    counts=torch.tensor([sum(row['label']==label for row in groups['train']) for label in labels],dtype=torch.float32)
    loss_fn=nn.CrossEntropyLoss(weight=1/counts)
    best=-1;best_loss=float('inf');stale=0;history=[];best_state=None;best_epoch=0
    for epoch in range(300):
        head.train();total=0
        for x,y in loaders['train']:
            optimizer.zero_grad();loss=loss_fn(head(x),y);loss.backward();optimizer.step();total+=float(loss.detach())
        val=evaluate(head,loaders['val'],len(labels));head.eval()
        with torch.no_grad():val_loss=sum(float(loss_fn(head(x),y)) for x,y in loaders['val'])/len(loaders['val'])
        history.append({'epoch':epoch+1,'loss':total/len(loaders['train']),'validation':val,'validationLoss':val_loss})
        improved=val['macroF1']>best+1e-8 or (abs(val['macroF1']-best)<1e-8 and val_loss<best_loss)
        if improved:
            best=val['macroF1'];best_loss=val_loss;best_epoch=epoch+1;best_state={k:v.detach().clone() for k,v in head.state_dict().items()};stale=0
        else:stale+=1
        if epoch%10==0:print(json.dumps(history[-1]),flush=True)
        if stale>=50:break
    head.load_state_dict(best_state);head.eval()
    model.classifier.load_state_dict(best_state);model.feature_mean.copy_(mean);model.feature_scale.copy_(scale);model.eval()
    validation=evaluate(head,loaders['val'],len(labels))
    if validation['macroF1']<=baseline['validation']['macroF1']:
        (out/'validation_only.json').write_text(json.dumps({'validation':validation,'baselineValidation':baseline['validation']},indent=2),encoding='utf-8')
        print(json.dumps({'installed':False,'reason':'Validation macro F1 did not improve; test and fresh audit remain unevaluated'}),flush=True);return
    audit_path=ROOT/'data/public_datasets/fresh_xray_audit/manifest.csv'
    if not audit_path.is_file():raise ValueError('Fresh audit download must complete before final evaluation.')
    # Final evaluations occur only after validation has selected the head and audit acquisition is complete.
    test=evaluate(head,loaders['test'],len(labels))
    audit_rows=csv_rows(audit_path);audit_hashes=set()
    for row in audit_rows:
        digest=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
        if digest!=row['sha256'] or digest in seen or digest in audit_hashes:raise ValueError('Audit checksum mismatch or image leakage')
        audit_hashes.add(digest)
    audit_features=(extract(audit_rows,'Fresh audit')-mean)/scale
    audit_loader=DataLoader(TensorDataset(audit_features,torch.tensor([labels.index(row['label']) for row in audit_rows])),batch_size=32)
    audit=evaluate(head,audit_loader,len(labels))
    metrics={'architecture':ARCHITECTURE,'labels':labels,'createdAt':datetime.now(timezone.utc).isoformat(),'splitUnit':'image','patientIndependent':False,'warning':WARNING,'trainingImages':len(groups['train']),'validationImages':len(groups['val']),'testImages':len(groups['test']),'freshAuditImages':len(audit_rows),'bestEpoch':best_epoch,'validation':validation,'test':test,'freshAudit':audit,'baselineValidation':baseline['validation'],'baselineTest':baseline.get('test'),'history':history,'selectionRule':'Highest validation macro F1, lower validation loss for ties; no test or audit tuning. Promotion requires validation macro F1 greater than the existing baseline.','pretrainedSource':'https://download.pytorch.org/models/resnet18-f37072fd.pth','featureNormalization':'Mean/std fitted on training images only; frozen pretrained backbone.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    tb=labels.index('Tuberculosis');n=sum(audit['confusionMatrix'][tb]);recall=audit['confusionMatrix'][tb][tb]/max(n,1)
    warning=WARNING+f" Fresh image-audit accuracy: {audit['accuracy']*100:.1f}% on {len(audit_rows)} images; TB recall: {recall*100:.1f}% on {n} images."
    payload={'architecture':ARCHITECTURE,'state':model.state_dict(),'labels':labels,'normal_label':'Normal','preprocessing':PREPROCESSING,'modelVersion':str(uuid.uuid4()),'validation':validation,'test':test,'freshAudit':audit,'researchWarning':warning,'evaluationScope':'image-level; patient independence not established','pretrainedSource':metrics['pretrainedSource']}
    torch.save(payload,out/'model.pt')
    if validation['macroF1']<=baseline['validation']['macroF1']:
        print(json.dumps({'installed':False,'reason':'Validation macro F1 did not improve','test':test,'freshAudit':audit}),flush=True);return
    backup=MODEL_DIR/'backups';backup.mkdir(exist_ok=True)
    shutil.copy2(MODEL_DIR/'xray.pt',backup/('xray_'+baseline['modelVersion']+'.pt'))
    shutil.copy2(MODEL_DIR/'xray_metrics.json',backup/('xray_'+baseline['modelVersion']+'_metrics.json'))
    pending=MODEL_DIR/'xray.pt.pending';torch.save(payload,pending);os.replace(pending,MODEL_DIR/'xray.pt')
    (MODEL_DIR/'xray_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps({'installed':True,'validation':validation,'test':test,'freshAudit':audit,'restartBackendRequired':True}),flush=True)

if __name__=='__main__':main()
