"""Download a reproducible real-data starter set; never generates patient data or predictions.
Run from the project root: python backend/fetch_datasets.py --per-class 20
Use --per-class 0 for all supported files (several GB). No Kaggle credentials required
by the public endpoints at the time of implementation. Access rules may change.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import random
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'public_datasets'
SOURCES = {
    'icbhi': ('vbookshelf/respiratory-sound-database', 2),
    'pneumonia': ('paultimothymooney/chest-xray-pneumonia', 2),
    'tb': ('tawsifurrahman/tuberculosis-tb-chest-xray-dataset', 3),
}
AUDIO_LABELS = {'Healthy': 'Normal Breathing', 'Asthma': 'Asthma', 'COPD': 'COPD',
                'Pneumonia': 'Pneumonia-related respiratory pattern'}


def request(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def index(source):
    slug, version = SOURCES[source]
    cache = DATA / 'metadata' / (source + '_index.json')
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))
    files, token, seen = [], '', set()
    while True:
        query = {'pageSize': 200, 'datasetVersionNumber': version}
        if token:
            query['pageToken'] = token
        result = json.loads(request('https://www.kaggle.com/api/v1/datasets/list/' + slug + '?' + urllib.parse.urlencode(query)))
        files.extend(result.get('datasetFiles', []))
        token = result.get('nextPageToken', '')
        print(source, 'indexed', len(files), flush=True)
        if not token:
            break
        if token in seen:
            raise RuntimeError('Repeated pagination token; refusing incomplete index')
        seen.add(token)
    cache.write_text(json.dumps(files, indent=2), encoding='utf-8')
    metadata = json.loads(request('https://www.kaggle.com/api/v1/datasets/view/' + slug))
    (cache.parent / (source + '_source.json')).write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return files


def fetch(source, name):
    slug, version = SOURCES[source]
    url = 'https://www.kaggle.com/api/v1/datasets/download/' + slug + '/' + urllib.parse.quote(name, safe='')
    content = request(url + '?datasetVersionNumber=' + str(version))
    if content.startswith(b'PK\x03\x04'):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if Path(name).suffix.lower() == '.xlsx' and '[Content_Types].xml' in archive.namelist():
                return content
            members = [i for i in archive.infolist() if not i.is_dir()]
            if len(members) != 1:
                raise ValueError('Expected one requested file in download')
            content = archive.read(members[0])
    return content


def candidates(source, files, diagnoses):
    found = {}
    for item in sorted(files, key=lambda item: item['name']):
        name = item['name']
        path = Path(name)
        if '__MACOSX' in name or path.name.startswith('.'):
            continue
        if source == 'icbhi':
            if path.suffix.lower() != '.wav':
                continue
            patient = path.stem.split('_')[0]
            original = diagnoses.get(patient)
            if original not in AUDIO_LABELS:
                continue
            label, modality = AUDIO_LABELS[original], 'audio'
            patient_id, basis = 'icbhi:' + patient, 'source diagnosis ID and recording filename'
        else:
            if path.suffix.lower() not in {'.jpeg', '.jpg', '.png'}:
                continue
            original = path.parent.name
            if original.lower() not in {'normal', 'pneumonia', 'tuberculosis'}:
                continue
            label, modality = original.capitalize(), 'xray'
            # These public image filenames do not establish verified patient identity.
            patient_id, basis = '', 'not verified; consult original patient metadata before splitting'
        found.setdefault((label, path.name.lower()), {
            'modality': modality, 'label': label, 'source_label': original,
            'source': source, 'source_file': name, 'patient_id': patient_id,
            'patient_id_basis': basis,
            'source_split': next((p for p in path.parts if p in {'train', 'val', 'test'}), '')})
    return list(found.values())


def choose(rows, limit):
    rng = random.Random(5785)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['patient_id'] or row['source_file']].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while keys and (not limit or len(selected) < limit):
        remaining = []
        for key in keys:
            if limit and len(selected) >= limit:
                break
            selected.append(groups[key].pop())
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-class', type=int, default=20)
    args = parser.parse_args()
    if args.per_class < 0:
        parser.error('--per-class must be nonnegative')
    (DATA / 'metadata').mkdir(parents=True, exist_ok=True)
    inventory, coverage = [], []
    for source in SOURCES:
        files = index(source)
        diagnoses = {}
        if source == 'icbhi':
            diagnosis_name = next(f['name'] for f in files if Path(f['name']).name == 'patient_diagnosis.csv')
            content = fetch(source, diagnosis_name)
            (DATA / 'metadata' / 'patient_diagnosis.csv').write_bytes(content)
            diagnoses = dict(csv.reader(io.StringIO(content.decode('utf-8-sig'))))
        rows = candidates(source, files, diagnoses)
        labels = sorted({row['label'] for row in rows})
        for label in labels:
            available = [r for r in rows if r['label'] == label]
            selected = choose(available, args.per_class)
            for number, row in enumerate(selected, 1):
                destination = DATA / row['modality'] / label / source / Path(row['source_file']).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    content = destination.read_bytes()
                else:
                    content = fetch(source, row['source_file'])
                    if row['modality'] == 'audio' and not content.startswith(b'RIFF'):
                        raise ValueError('Expected WAV: ' + row['source_file'])
                    if row['modality'] == 'xray' and not content.startswith((b'\x89PNG', b'\xff\xd8')):
                        raise ValueError('Expected PNG/JPEG: ' + row['source_file'])
                    destination.write_bytes(content)
                row.update(path=destination.relative_to(ROOT).as_posix(), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
                inventory.append(row)
                print(source, label, str(number) + '/' + str(len(selected)), flush=True)
            coverage.append({'source': source, 'label': label, 'source_unique_filenames': len(available),
                             'downloaded': len(selected),
                             'patients': len({r['patient_id'] for r in selected if r['patient_id']}) if source == 'icbhi' else None})
    with (DATA / 'inventory.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['modality','label','source_label','source','source_file','patient_id','patient_id_basis','source_split','path','bytes','sha256'])
        writer.writeheader()
        writer.writerows(inventory)
    report = {'per_class_per_source': args.per_class, 'coverage': coverage,
              'total_downloaded': len(inventory), 'training_ready': False,
              'notes': ['Starter subset unless --per-class 0 was used.',
                        'Source normal labels are dataset controls, not proof of overall health.',
                        'X-ray patient IDs require curation before patient-disjoint splits.',
                        'ICBHI asthma has insufficient distinct patients for train/val/test.',
                        'The two modalities are independent cohorts, not paired patient tests.',
                        'No model training or diagnostic confidence values are generated.']}
    (DATA / 'coverage.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
