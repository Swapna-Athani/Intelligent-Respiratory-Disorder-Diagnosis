"""Integration checks for the installed real X-ray transfer model; no remote writes."""
import csv,io,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from app import app
from inference import predict
from config import MODEL_DIR

class TransferModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        metadata=MODEL_DIR/'xray_metrics.json'
        if not metadata.exists() or json.loads(metadata.read_text()) .get('architecture')!='resnet18-transfer-v1':raise unittest.SkipTest('Transfer model not installed')
        manifest=Path(__file__).resolve().parent/'data/public_datasets/fresh_xray_audit/manifest.csv'
        with manifest.open(newline='',encoding='utf-8') as handle:cls.rows=list(csv.DictReader(handle))
        cls.image=Path(cls.rows[0]['path']).read_bytes()
    def test_same_pixels_ignore_folder_and_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            a=folder/'Normal'/'healthy.png';b=folder/'Tuberculosis'/'disease.png'
            for path in (a,b):path.parent.mkdir();path.write_bytes(self.image)
            first=predict('xray',a);second=predict('xray',b)
            self.assertEqual(first['prediction'],second['prediction'])
            self.assertEqual(first['scores'],second['scores'])
            self.assertEqual(first['modelVersion'],second['modelVersion'])
            self.assertEqual(set(first['scores']),{'Normal','Pneumonia','Tuberculosis'})
    def test_upload_api_uses_pixels_instead_of_upload_name(self):
        def database(method,*args):return {'sampleId':'001'} if method=='GET' else None
        with patch('app.identity',return_value='engineering-test'),patch('app.database',side_effect=database):
            results=[]
            for name in ('Normal.png','Tuberculosis.png'):
                response=app.test_client().post('/api/analyze/xray',data={'file':(io.BytesIO(self.image),name),'testId':'00000000-0000-0000-0000-000000000000'},headers={'Authorization':'Bearer engineering-test'})
                self.assertEqual(response.status_code,200,response.json)
                self.assertTrue(response.json['saved'])
                results.append(response.json['result'])
            self.assertEqual(results[0]['scores'],results[1]['scores'])
            self.assertIn('patient',results[0]['evaluationScope'])
    def test_audit_has_no_exact_training_or_validation_duplicates(self):
        manifest=MODEL_DIR/'xray_prototype_run/image_split_manifest.csv'
        with manifest.open(newline='',encoding='utf-8') as handle:known={row['sha256'] for row in csv.DictReader(handle)}
        hashes=[row['sha256'] for row in self.rows]
        self.assertEqual(len(hashes),60)
        self.assertEqual(len(set(hashes)),60)
        self.assertFalse(set(hashes)&known)

if __name__=='__main__':unittest.main()
