"""Engineering-only in-memory fixtures. No fabricated results are shipped or saved."""
import io,tempfile,unittest,csv
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from processing import load_audio,clean_audio,audio_tensor,audio_preview,load_xray,xray_tensor
from model import ScreeningCNN
from inference import ModelUnavailable,predict
from train import read_manifest
from app import app
class BackendTests(unittest.TestCase):
    def wav(self):
        out=io.BytesIO();t=np.arange(32000)/16000;sf.write(out,(.2*np.sin(2*np.pi*300*t)).astype('float32'),16000,format='WAV');out.seek(0);return out
    def test_real_preprocessing_shapes(self):
        y=clean_audio(load_audio(self.wav()));x=audio_tensor(y)
        self.assertEqual(tuple(x.shape),(2,64,128));self.assertTrue(torch.isfinite(x).all());self.assertEqual(len(audio_preview(y)['waveform']),1200)
        out=io.BytesIO();Image.new('L',(128,128),128).save(out,format='PNG');out.seek(0)
        self.assertEqual(tuple(xray_tensor(load_xray(out)).shape),(1,224,224))
    def test_cnn_backward(self):
        for channels,h,w in [(2,64,128),(1,224,224)]:
            m=ScreeningCNN(channels,2);loss=torch.nn.CrossEntropyLoss()(m(torch.zeros((2,channels,h,w))),torch.tensor([0,1]));loss.backward();self.assertIsNotNone(m.features[0].weight.grad)
    def test_missing_models_never_create_predictions(self):
        with tempfile.TemporaryDirectory() as d,patch('inference.MODEL_DIR',Path(d)):
            with self.assertRaises(ModelUnavailable):predict('audio',self.wav())
    def test_bad_audio_is_rejected(self):
        with self.assertRaises(ValueError):load_audio(io.BytesIO(b'not audio'))
    def test_auth_is_required(self):
        c=app.test_client();self.assertEqual(c.post('/api/preview/audio').status_code,401)
        self.assertEqual(c.get('/api/health').status_code,200)
    def test_preview_and_missing_model_api(self):
        c=app.test_client()
        with patch('app.identity',return_value='test-user'),patch('app.database',return_value={'sampleId':'test'}) as db, tempfile.TemporaryDirectory() as d,patch('inference.MODEL_DIR',Path(d)):
            res=c.post('/api/preview/audio',data={'file':(self.wav(),'test.wav')},headers={'Authorization':'Bearer unit-test'})
            self.assertEqual(res.status_code,200);self.assertIn('spectrogram',res.json)
            res=c.post('/api/analyze/audio',data={'file':(self.wav(),'test.wav'),'testId':'00000000-0000-0000-0000-000000000000'},headers={'Authorization':'Bearer unit-test'})
            self.assertEqual(res.status_code,503)
            self.assertTrue(all(call.args[0]=='GET' for call in db.call_args_list))
    def test_xray_accepts_sample_id_one_with_internal_test_uuid(self):
        c=app.test_client();image=io.BytesIO();Image.new('L',(128,128),128).save(image,format='PNG');image.seek(0)
        with patch('app.identity',return_value='test-user'),patch('app.database',return_value={'sampleId':'1'}) as db,tempfile.TemporaryDirectory() as d,patch('inference.MODEL_DIR',Path(d)):
            response=c.post('/api/analyze/xray',data={'file':(image,'sample-1.png'),'testId':'00000000-0000-0000-0000-000000000000'},headers={'Authorization':'Bearer unit-test'})
            self.assertEqual(response.status_code,503)
            self.assertIn('model',response.json['error'].lower())
            self.assertEqual(db.call_args.args[3],'tests/00000000-0000-0000-0000-000000000000')
            self.assertTrue(all(call.args[0]=='GET' for call in db.call_args_list))
    def test_health_reports_loaded_model_compatibility(self):
        from inference import model_status
        from processing import XRAY_VERSION
        with tempfile.TemporaryDirectory() as directory,patch('inference.MODEL_DIR',Path(directory)):
            path=Path(directory)/'xray.pt'
            payload={'state':ScreeningCNN(1,2).state_dict(),'labels':['Normal','Tuberculosis'],'normal_label':'Normal','preprocessing':'incompatible-version','modelVersion':'engineering-test'}
            torch.save(payload,path)
            status=model_status('xray')
            self.assertFalse(status['ready']);self.assertIn('incompatible',status['reason'])
            response=app.test_client().get('/api/health')
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json['backendVersion'],'ks5785-transfer-v1')
            self.assertFalse(response.json['models']['xray'])
            self.assertIn('incompatible',response.json['modelDetails']['xray']['reason'])
            payload['preprocessing']=XRAY_VERSION
            replacement=Path(directory)/'replacement.pt';torch.save(payload,replacement);replacement.replace(path)
            self.assertTrue(model_status('xray')['ready'])
    def test_report_escapes_patient_content_as_plain_text(self):
        c=app.test_client();r={'sampleId':'Patient (untrusted) <tag>','createdAt':'2026-10-27','audio':{'prediction':'Normal','confidence':75.0,'normal':True,'analyzedAt':'2026-10-27','modelVersion':'test'}}
        with patch('app.identity',return_value='test-user'),patch('app.database',return_value=r):
            response=c.get('/api/report/00000000-0000-0000-0000-000000000000',headers={'Authorization':'Bearer unit-test'})
            self.assertEqual(response.status_code,200);self.assertTrue(response.data.startswith(b'%PDF'))
    def test_patient_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'a').write_bytes(b'a');(p/'b').write_bytes(b'b');(p/'manifest.csv').write_text('path,label,patient_id,split\na,Normal,p1,train\nb,Asthma,p1,val\n')
            with self.assertRaisesRegex(ValueError,'Patient leakage'):read_manifest(p/'manifest.csv')
if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
