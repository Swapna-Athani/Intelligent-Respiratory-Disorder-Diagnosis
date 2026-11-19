"""Verify downloaded files and export measured image/audio values (no predictions)."""
import collections
import csv
import hashlib
import json
import wave
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'public_datasets'


def main():
    with (DATA / 'inventory.csv').open(newline='', encoding='utf-8') as handle:
        records = list(csv.DictReader(handle))
    values, hashes, problems = [], {}, []
    for record in records:
        path = (ROOT / record['path']).resolve()
        if not path.is_relative_to(DATA.resolve()):
            raise ValueError('Inventory path outside dataset folder')
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != record['sha256'] or len(content) != int(record['bytes']):
            raise ValueError('File content mismatch: ' + record['path'])
        if digest in hashes:
            problems.append({'issue': 'identical bytes', 'first': hashes[digest], 'second': record['path']})
        hashes[digest] = record['path']
        value = {key: record[key] for key in ['path', 'modality', 'label', 'source', 'patient_id']}
        if record['modality'] == 'audio':
            with wave.open(str(path), 'rb') as audio:
                channels, width, rate, frames = audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()
                raw = audio.readframes(frames)
            if len(raw) != frames * channels * width or frames == 0:
                raise ValueError('Incomplete/empty WAV: ' + str(path))
            if width == 2:
                signal = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.0
            elif width == 1:
                signal = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
            elif width == 4:
                signal = np.frombuffer(raw, dtype='<i4').astype(np.float64) / 2147483648.0
            elif width == 3:
                packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
                pcm = packed[:, 0] | packed[:, 1] << 8 | packed[:, 2] << 16
                signal = ((pcm ^ 0x800000) - 0x800000).astype(np.float64) / 8388608.0
            else:
                raise ValueError('Unsupported PCM sample width: ' + str(width))
            value.update(sample_rate_hz=rate, channels=channels, bits_per_sample=width * 8,
                         frames=frames, duration_seconds=frames / rate,
                         peak_amplitude=float(np.abs(signal).max()), rms_amplitude=float(np.sqrt(np.mean(signal ** 2))))
        else:
            with Image.open(path) as image:
                image.load()
                pixels = np.asarray(image.convert('L'))
                value.update(width_pixels=image.width, height_pixels=image.height,
                             image_mode=image.mode, pixel_min=int(pixels.min()),
                             pixel_max=int(pixels.max()), pixel_mean=float(pixels.mean()))
        values.append(value)
    fields = ['path','modality','label','source','patient_id','sample_rate_hz','channels','bits_per_sample','frames','duration_seconds','peak_amplitude','rms_amplitude','width_pixels','height_pixels','image_mode','pixel_min','pixel_max','pixel_mean']
    with (DATA / 'file_values.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
    summary = {'validated_files': len(values), 'duplicate_files': problems,
               'counts': dict(collections.Counter(v['modality'] + ': ' + v['label'] for v in values)),
               'normal_means': 'Source Normal/Healthy label only; not overall clinical health.',
               'audio_values': 'PCM amplitudes normalized to digital full scale; not calibrated sound pressure.',
               'xray_values': '8-bit grayscale pixels; not Hounsfield units or disease severity.',
               'training': 'No training performed; inventory is not a patient-disjoint training manifest.'}
    (DATA / 'validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
