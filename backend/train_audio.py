"""Train a patient-disjoint respiratory audio prototype on genuine ICBHI labels.
Asthma is excluded: one source participant cannot support independent splits.
Validation/test use the exact recording-window averaging used by inference.
"""
import argparse,collections,csv,hashlib,json,os,random,shutil,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from config import MODEL_DIR
from model import ScreeningCNN
from processing import load_audio,clean_audio,audio_tensor,SR,SECONDS,AUDIO_VERSION
from train import read_manifest

ROOT=Path(__file__).resolve().parent
SUPPORTED={'Normal Breathing','COPD','Pneumonia-related respiratory pattern'}
WARNING='Exploratory ICBHI lung-sound prototype; not clinically validated. Asthma and other untrained conditions are unsupported. Recording-device and population differences can affect predictions. Scores are not disease probabilities. Do not use for medical decisions.'


def scores(matrix):
    matrix=np.asarray(matrix,dtype=np.int64);recalls=[];f1=[]
    for index in range(len(matrix)):
        tp=int(matrix[index,index]);recall=tp/max(int(matrix[index].sum()),1);precision=tp/max(int(matrix[:,index].sum()),1)
        recalls.append(recall);f1.append(2*precision*recall/max(precision+recall,1e-12))
    return {'accuracy':float(np.trace(matrix)/max(int(matrix.sum()),1)),'balancedAccuracy':sum(recalls)/len(recalls),'macroF1':sum(f1)/len(f1),'confusionMatrix':matrix.tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--epochs',type=int,default=60);args=parser.parse_args()
    if args.epochs<1:parser.error('epochs must be positive')
    torch.set_num_threads(2);torch.manual_seed(5799);np.random.seed(5799);rng=random.Random(5799)
    with (ROOT/'data/public_datasets/inventory.csv').open(newline='',encoding='utf-8') as handle:inventory=list(csv.DictReader(handle))
    with (ROOT/'data/public_datasets/metadata/patient_diagnosis.csv').open(newline='',encoding='utf-8-sig') as handle:diagnoses=dict(csv.reader(handle))
    mapping={'Healthy':'Normal Breathing','COPD':'COPD','Pneumonia':'Pneumonia-related respiratory pattern'}
    rows=[];excluded=[];patient_labels={};hashes=set()
    for row in inventory:
        if row['modality']!='audio':continue
        if row['label'] not in SUPPORTED:excluded.append({'path':row['path'],'label':row['label'],'reason':'insufficient patients for independent splits'});continue
        patient=Path(row['path']).stem.split('_')[0]
        if row['patient_id']!='icbhi:'+patient or mapping.get(diagnoses.get(patient))!=row['label']:raise ValueError('Audio diagnosis/patient annotation mismatch')
        path=(ROOT/row['path']).resolve();digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row['sha256'] or digest in hashes:raise ValueError('Checksum mismatch or duplicate audio file')
        hashes.add(digest);patient_labels[row['patient_id']]=row['label']
        rows.append({'path':str(path),'label':row['label'],'patient_id':row['patient_id'],'sha256':digest})
    assignment={}
    for label in sorted(SUPPORTED):
        subjects=sorted(patient for patient,diagnosis in patient_labels.items() if diagnosis==label);rng.shuffle(subjects)
        if len(subjects)<5:raise ValueError('Need at least five subjects in every supported class')
        ntest=max(1,round(len(subjects)*.2));nval=max(1,round(len(subjects)*.2))
        for index,patient in enumerate(subjects):assignment[patient]='test' if index<ntest else 'val' if index<ntest+nval else 'train'
    manifest=ROOT/'datasets/audio_manifest.csv'
    if manifest.exists() and len(manifest.read_text(encoding='utf-8-sig').strip().splitlines())>1:
        shutil.copy2(manifest,manifest.with_name('audio_manifest.previous.csv'))
    with manifest.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['path','label','patient_id','split']);writer.writeheader()
        for row in rows:writer.writerow({'path':os.path.relpath(row['path'],manifest.parent),'label':row['label'],'patient_id':row['patient_id'],'split':assignment[row['patient_id']]})
    validated,labels=read_manifest(manifest)
    for row in rows:row['split']=assignment[row['patient_id']]
    out=MODEL_DIR/'audio_run';out.mkdir(parents=True,exist_ok=True)
    counts={split:{'recordings':dict(collections.Counter(r['label'] for r in rows if r['split']==split)),'patients':dict(collections.Counter(label for patient,label in patient_labels.items() if assignment[patient]==split))} for split in ('train','val','test')}
    preparation={'counts':counts,'excluded':excluded,'splitUnit':'patient','source':'ICBHI diagnosis table and original recording filenames','seed':5799,'labels':labels}
    (out/'preparation.json').write_text(json.dumps(preparation,indent=2),encoding='utf-8')
    print(json.dumps(preparation),flush=True)
    cache=out/'features';cache.mkdir(exist_ok=True);version=hashlib.sha256(AUDIO_VERSION.encode()).hexdigest()[:12]
    windows={}
    for index,row in enumerate(rows,1):
        path=cache/(row['sha256']+'_'+version+'.pt')
        if path.exists():value=torch.load(path,map_location='cpu',weights_only=True)
        else:
            y=clean_audio(load_audio(row['path']))
            value=torch.stack([audio_tensor(y[start:start+SR*SECONDS]) for start in range(0,len(y),SR*SECONDS) if len(y[start:start+SR*SECONDS])>=SR])
            torch.save(value,path)
        if not len(value):raise ValueError('No valid audio windows')
        windows[row['path']]=value
        if index%10==0 or index==len(rows):print(f'Audio features {index}/{len(rows)}',flush=True)
    training=[row for row in rows if row['split']=='train']
    x=torch.cat([windows[row['path']] for row in training]);y=torch.cat([torch.full((len(windows[row['path']]),),labels.index(row['label']),dtype=torch.long) for row in training])
    loader=DataLoader(TensorDataset(x,y),batch_size=16,shuffle=True)
    counts_tensor=torch.bincount(y,minlength=len(labels)).float()
    model=ScreeningCNN(2,len(labels));optimizer=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
    loss_fn=nn.CrossEntropyLoss(weight=1/counts_tensor)
    def evaluate(split):
        model.eval();matrix=np.zeros((len(labels),len(labels)),dtype=np.int64);subjects=collections.defaultdict(list);truth={};losses=[]
        with torch.inference_mode():
            for row in rows:
                if row['split']!=split:continue
                pieces=windows[row['path']];prob=torch.cat([model(batch).softmax(1) for batch in pieces.split(16)]).mean(0)
                target=labels.index(row['label']);matrix[target,int(prob.argmax())]+=1;losses.append(float(-prob[target].clamp_min(1e-12).log()))
                subjects[row['patient_id']].append(prob);truth[row['patient_id']]=target
        patient_matrix=np.zeros_like(matrix)
        for patient,values in subjects.items():patient_matrix[truth[patient],int(torch.stack(values).mean(0).argmax())]+=1
        return {'recording':scores(matrix),'patient':scores(patient_matrix),'recordingLoss':sum(losses)/len(losses)}
    best=-1;best_loss=float('inf');stale=0;history=[];candidate=out/'best.pt';best_epoch=0
    for epoch in range(args.epochs):
        model.train();total=0
        for features,target in loader:
            optimizer.zero_grad();loss=loss_fn(model(features),target);loss.backward();optimizer.step();total+=float(loss.detach())
        validation=evaluate('val');history.append({'epoch':epoch+1,'loss':total/len(loader),'validation':validation});print(json.dumps(history[-1]),flush=True)
        score=validation['patient']['macroF1'];value=validation['recordingLoss']
        if score>best+1e-8 or (abs(score-best)<1e-8 and value<best_loss):
            best=score;best_loss=value;stale=0;best_epoch=epoch+1
            torch.save({'architecture':'screening-cnn','state':model.state_dict(),'labels':labels,'normal_label':'Normal Breathing','preprocessing':AUDIO_VERSION,'modelVersion':str(uuid.uuid4()),'validation':validation['recording'],'patientValidation':validation['patient'],'researchWarning':WARNING,'evaluationScope':'patient-disjoint within ICBHI; same-source prototype, not clinically validated','unsupportedLabels':['Asthma'],'counts':counts},candidate)
        else:stale+=1
        if stale>=15:break
    payload=torch.load(candidate,map_location='cpu',weights_only=True);model.load_state_dict(payload['state']);test=evaluate('test')
    metrics={'labels':labels,'createdAt':datetime.now(timezone.utc).isoformat(),'splitUnit':'patient','counts':counts,'excluded':excluded,'bestEpoch':best_epoch,'history':history,'test':test,'evaluationProtocol':'Each recording is split into consecutive 5-second windows, last pieces below one second omitted, softmax averaged exactly as inference. Patient metrics additionally average recording scores per subject.','warning':WARNING,'selectionRule':'Validation patient macro F1, recording negative log-likelihood breaks ties; test evaluated only after checkpoint selection.'}
    payload['test']=test
    nrecord=sum(sum(row) for row in test['recording']['confusionMatrix']);npatient=sum(sum(row) for row in test['patient']['confusionMatrix'])
    payload['researchWarning']+=f" Held-out same-source recording accuracy {test['recording']['accuracy']*100:.1f}% on {nrecord} recordings from {npatient} subjects; very few pneumonia subjects limit evaluation."
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (MODEL_DIR/'audio_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    installed=MODEL_DIR/'audio.pt'
    if installed.exists():
        backup=MODEL_DIR/'backups';backup.mkdir(exist_ok=True);shutil.copy2(installed,backup/('audio_before_'+datetime.now().strftime('%Y%m%d%H%M%S')+'.pt'))
    pending=MODEL_DIR/'audio.pt.pending';torch.save(payload,pending);os.replace(pending,installed)
    print(json.dumps({'installed':str(installed),'labels':labels,'bestEpoch':best_epoch,'test':test}),flush=True)

if __name__=='__main__':main()
