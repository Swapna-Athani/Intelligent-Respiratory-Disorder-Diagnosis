"""Installed-audio-model integration checks; no remote patient writes."""
import csv,io,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from config import MODEL_DIR
from inference import load_model,predict,model_status
from train import read_manifest
from app import app

class AudioModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        if not (MODEL_DIR/'audio.pt').exists():raise unittest.SkipTest('Audio model not installed')
        cls.rows,cls.labels=read_manifest(Path(__file__).resolve().parent/'datasets/audio_manifest.csv')
        cls.record=min((row for row in cls.rows if row['split']=='test'),key=lambda row:Path(row['path']).stat().st_size)
        cls.raw=Path(cls.record['path']).read_bytes()
    def test_patient_groups_are_disjoint_and_asthma_is_excluded(self):
        groups={split:{row['patient_id'] for row in self.rows if row['split']==split} for split in ('train','val','test')}
        self.assertFalse(groups['train']&groups['val']);self.assertFalse(groups['train']&groups['test']);self.assertFalse(groups['val']&groups['test'])
        self.assertEqual(set(self.labels),{'Normal Breathing','COPD','Pneumonia-related respiratory pattern'})
        self.assertTrue(all(row['patient_id'].startswith('icbhi:') for row in self.rows))
        status=model_status('audio');self.assertTrue(status['ready']);self.assertIn('Asthma',status['unsupportedLabels'])
    def test_real_recording_scores_ignore_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'Normal.wav';b=Path(directory)/'Asthma.wav';a.write_bytes(self.raw);b.write_bytes(self.raw)
            first=predict('audio',a);second=predict('audio',b)
            self.assertEqual(first['scores'],second['scores'])
            self.assertEqual(set(first['scores']),set(self.labels))
            self.assertIn('Asthma',first['note']);self.assertIn('patient-disjoint',first['evaluationScope'])
    def test_audio_upload_api_uses_real_model(self):
        def database(method,*args):return {'sampleId':'001'} if method=='GET' else None
        with patch('app.identity',return_value='engineering-test'),patch('app.database',side_effect=database):
            response=app.test_client().post('/api/analyze/audio',data={'file':(io.BytesIO(self.raw),'uploaded-audio.wav'),'testId':'00000000-0000-0000-0000-000000000000'},headers={'Authorization':'Bearer engineering-test'})
            self.assertEqual(response.status_code,200,response.json)
            self.assertTrue(response.json['saved'])
            self.assertEqual(set(response.json['result']['scores']),set(self.labels))
            self.assertIn('spectrogram',response.json['result']['visualization'])

if __name__=='__main__':unittest.main()
