from datetime import datetime,timezone
from functools import lru_cache
import numpy as np
import torch
from config import MODEL_DIR
from model import ScreeningCNN
from processing import load_audio,clean_audio,audio_tensor,load_xray,xray_tensor,SR,SECONDS,AUDIO_VERSION,XRAY_VERSION
class ModelUnavailable(Exception):pass

def load_model(kind):
    path=MODEL_DIR/f'{kind}.pt'
    if not path.is_file(): raise ModelUnavailable(f'{kind.capitalize()} model is not trained. Train it before analysis.')
    stat=path.stat()
    return _load_model_file(kind,str(path.resolve()),stat.st_mtime_ns,stat.st_size)

@lru_cache(maxsize=4)
def _load_model_file(kind,path,modified,size):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    architecture=payload.get('architecture','screening-cnn')
    if architecture=='resnet18-transfer-v1' and kind=='xray':
        from xray_transfer import XrayTransferModel,PREPROCESSING
        expected=PREPROCESSING
        model=XrayTransferModel(len(payload['labels']))
    elif architecture=='screening-cnn':
        expected=AUDIO_VERSION if kind=='audio' else XRAY_VERSION
        model=ScreeningCNN(2 if kind=='audio' else 1,len(payload['labels']))
    else:raise ModelUnavailable('Unsupported model architecture; update the backend.')
    if payload.get('preprocessing')!=expected:raise ModelUnavailable(f'Model preprocessing is incompatible: checkpoint uses {payload.get("preprocessing")}, backend expects {expected}. Restart the updated project backend or install matching model weights.')
    model.load_state_dict(payload['state']);model.eval();return model,payload


def model_status(kind):
    try:
        _,meta=load_model(kind)
        return {'ready':True,'architecture':meta.get('architecture','screening-cnn'),'preprocessing':meta['preprocessing'],'modelVersion':meta['modelVersion'],'labels':meta['labels'],'unsupportedLabels':meta.get('unsupportedLabels',[])}
    except Exception as error:
        if isinstance(error,ModelUnavailable):reason=str(error)
        elif isinstance(error,ModuleNotFoundError):reason='A model dependency is missing. Install backend/requirements.txt in the project virtual environment and restart the backend.'
        else:reason='The model checkpoint could not be loaded. Check the backend dependencies and matching model weights.'
        return {'ready':False,'reason':reason}

def predict(kind,source):
    model,meta=load_model(kind)
    if kind=='audio':
        y=clean_audio(load_audio(source))
        windows=[audio_tensor(y[i:i+SR*SECONDS]) for i in range(0,len(y),SR*SECONDS) if len(y[i:i+SR*SECONDS])>=SR]
        with torch.inference_mode():prob=model(torch.stack(windows)).softmax(1).mean(0).numpy()
    else:
        image=load_xray(source)
        if meta.get('architecture')=='resnet18-transfer-v1':
            from xray_transfer import transfer_tensor
            tensor=transfer_tensor(image)
        else:tensor=xray_tensor(image)
        with torch.inference_mode():prob=model(tensor[None]).softmax(1)[0].numpy()
    index=int(np.argmax(prob));label=meta['labels'][index]
    return {'prediction':label,'confidence':round(float(prob[index])*100,2),'normal':label==meta['normal_label'],'scores':{k:round(float(v)*100,2) for k,v in zip(meta['labels'],prob)},'modelVersion':meta['modelVersion'],'analyzedAt':datetime.now(timezone.utc).isoformat(),'note':meta.get('researchWarning','Uncalibrated model score; preliminary screening only.'),'evaluationScope':meta.get('evaluationScope','See model training records.')}
