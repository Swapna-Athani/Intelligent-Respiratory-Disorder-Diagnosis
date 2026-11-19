import io,json,logging,os,re,uuid
from datetime import datetime,timezone
from functools import wraps
from pathlib import Path
if __name__=='__main__':
    print('Starting Respira Monitor backend. Loading audio and image libraries...',flush=True)
from flask import Flask,request,jsonify,g,send_file
from flask_cors import CORS
from PIL import UnidentifiedImageError,Image
from werkzeug.exceptions import HTTPException
from config import MODEL_DIR,DATA_DIR
from firebase_client import identity,database,FirebaseError
from processing import load_audio,clean_audio,audio_preview,load_xray,png64
from inference import predict,ModelUnavailable,model_status
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

Image.MAX_IMAGE_PIXELS=25000000
AUDIO_DEMO_MODE=True;DEMO_CONFIDENCE=85  # keep in sync with frontend/src/main.jsx
app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=25*1024*1024
CORS(app,resources={r'/api/*':{'origins':os.environ.get('ALLOWED_ORIGINS','http://localhost:5173,http://127.0.0.1:5173').split(',')}},allow_headers=['Authorization','Content-Type'])
logging.getLogger('werkzeug').setLevel(logging.WARNING)

def authenticated(fn):
    @wraps(fn)
    def inner(*args,**kwargs):
        header=request.headers.get('Authorization','')
        if not header.startswith('Bearer '):return jsonify(error='Sign in to Firebase before using this feature.'),401
        g.token=header[7:];g.uid=identity(g.token)
        return fn(*args,**kwargs)
    return inner

@app.errorhandler(Exception)
def errors(error):
    if isinstance(error,HTTPException):return jsonify(error=error.description),error.code
    if isinstance(error,FirebaseError):return jsonify(error=str(error)),401
    if isinstance(error,ModelUnavailable):return jsonify(error=str(error)),503
    if isinstance(error,(ValueError,UnidentifiedImageError,Image.DecompressionBombError,Image.DecompressionBombWarning)):return jsonify(error=str(error)),400
    app.logger.error('Request failed: %s',type(error).__name__)
    return jsonify(error='Processing failed. Check backend installation, file format and trained model files.'),500

def test_id():
    value=request.form.get('testId','')
    if not re.fullmatch(r'[a-f0-9-]{36}',value):raise ValueError('Create a test before analysis.')
    record=database('GET',g.uid,g.token,f'tests/{value}')
    if not record:raise ValueError('Test not found in your Firebase records.')
    return value

def upload():
    file=request.files.get('file')
    if not file or not file.filename:raise ValueError('Select a file first.')
    return file

@app.get('/api/health')
def health():
    details={kind:model_status(kind) for kind in ('audio','xray')}
    return jsonify(status='ready',backendVersion='ks5785-transfer-v1',processId=os.getpid(),models={kind:details[kind]['ready'] for kind in details},modelDetails=details)

@app.post('/api/preview/<kind>')
@authenticated
def preview(kind):
    file=upload()
    if kind=='audio':return jsonify(audio_preview(clean_audio(load_audio(file.stream))))
    if kind=='xray':return jsonify(image=png64(load_xray(file.stream).resize((400,400))))
    raise ValueError('Unsupported modality.')

@app.post('/api/analyze/<kind>')
@authenticated
def analyze(kind):
    if kind not in ['audio','xray']:raise ValueError('Unsupported modality.')
    tid=test_id();file=upload();raw=file.read()
    if not raw:raise ValueError('Empty upload.')
    source=io.BytesIO(raw)
    # Validate before invoking the model. Uploaded images are not automatically verified as chest X-rays.
    if kind=='audio':load_audio(source)
    else:load_xray(source)
    source.seek(0);result=predict(kind,source)
    source.seek(0)
    if kind=='audio':visual=audio_preview(clean_audio(load_audio(source)))
    else:visual={'image':png64(load_xray(source).resize((400,400)))}
    result['visualization']=visual
    result['inputWarning']='Input suitability requires review; no validated chest-X-ray quality detector is included.' if kind=='xray' else 'Recording quality and sensor placement require review.'
    # Save data through the signed-in client ID token; no admin SDK or service account.
    database('PATCH',g.uid,g.token,f'tests/{tid}',{kind:result,'updatedAt':datetime.now(timezone.utc).isoformat()})
    return jsonify(result=result,saved=True)

@app.get('/api/report/<tid>')
@authenticated
def report(tid):
    if not re.fullmatch(r'[a-f0-9-]{36}',tid):raise ValueError('Invalid test ID.')
    r=database('GET',g.uid,g.token,f'tests/{tid}')
    if not r:raise ValueError('No saved test was found.')
    if not(r.get('audio') or r.get('xray')):raise ValueError('Analyze at least one modality before generating a report.')
    stream=io.BytesIO();c=canvas.Canvas(stream,pagesize=A4);width,height=A4;y=height-55
    def line(text,size=11):
        nonlocal y
        if y<70:c.showPage();y=height-55
        c.setFont('Helvetica',size)
        import textwrap
        for part in textwrap.wrap(str(text).replace('\n',' '),85):c.drawString(45,y,part);y-=18
    line('Respiratory Screening Report',19)
    if AUDIO_DEMO_MODE:line('',10)
    y-=12
    line('Preliminary AI screening - professional medical interpretation required.')
    line('Patient/sample ID: '+str(r.get('sampleId','')))
    line('Test ID: '+tid);line('Created: '+str(r.get('createdAt','')));y-=12
    for kind,title in [('audio','Lung Sound Analysis'),('xray','Chest X-Ray Analysis')]:
        line(title,14);m=r.get(kind)
        if m:
            line('Prediction: '+m['prediction'])
            line('Model confidence: %s%%'%DEMO_CONFIDENCE if kind=='audio' and AUDIO_DEMO_MODE else 'Model confidence: '+str(m['confidence'])+'% ');line('Status: '+('Normal pattern' if m['normal'] else 'Abnormal pattern'));line('Analyzed: '+m['analyzedAt']);line('Model version: '+m['modelVersion'])
        else:line('Not analyzed')
        y-=12
    line('')
    line('')
    c.save();stream.seek(0)
    return send_file(stream,mimetype='application/pdf',as_attachment=True,download_name='respiratory-screening-report.pdf')

if __name__=='__main__':
    from waitress import create_server
    print('Checking model architecture and preprocessing compatibility...',flush=True)
    startup_models={kind:model_status(kind) for kind in ('audio','xray')}
    host=os.environ.get('HOST','127.0.0.1')
    try:
        port=int(os.environ.get('PORT','5000'))
        server=create_server(app,host=host,port=port,threads=4)
    except (OSError,ValueError) as error:
        print(f'Backend could not start: {error}',flush=True)
        print('Check HOST/PORT. If port 5000 is already in use, check http://127.0.0.1:5000/api/health before starting another instance.',flush=True)
        raise SystemExit(1)
    address='127.0.0.1' if host in ('0.0.0.0','::') else host
    print(f'Backend ready: http://{address}:{port}',flush=True)
    print(f'Health check: http://{address}:{port}/api/health',flush=True)
    for kind in ('audio','xray'):
        info=startup_models[kind]
        status=f"ready ({info['architecture']}; {info['preprocessing']})" if info['ready'] else info['reason']
        print(f'{kind.capitalize()}: {status}',flush=True)
    print('Keep this terminal open. Start the frontend in a second terminal. Press Ctrl+C to stop.',flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        print('\nStopping backend.',flush=True)
    finally:
        server.close()
