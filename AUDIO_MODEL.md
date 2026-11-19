# Respiratory audio model

The audio model is trained from genuine ICBHI recordings with source patient diagnosis annotations. Its supported labels are Normal Breathing, COPD and Pneumonia-related respiratory pattern. Asthma is deliberately unsupported: the downloaded ICBHI release has one asthma participant/recording, which cannot support independent training/validation/test groups. The source Asthma file remains available as a dataset example; it is not used to manufacture a trained class.

The manifest is generated in `backend/datasets/audio_manifest.csv` from the real inventory. The preparation script verifies source patient IDs against the diagnosis table, validates source file hashes and assigns complete patients to a single split. The existing manifest leakage/duplicate checks are applied before training.

## Data and evaluation protocol

There are 172 recordings from 96 participants in the three supported classes:

| Split | Recordings | Participants | Pneumonia participants |
|---|---:|---:|---:|
| Training | 97 | 58 | 4 |
| Validation | 40 | 19 | 1 |
| Test | 35 | 19 | 1 |

The shared pipeline resamples to 16 kHz, filters/normalizes the signal, and extracts log-Mel and MFCC features. Recordings are divided into consecutive 5-second windows, padding shorter valid windows and omitting trailing pieces below one second. Validation/test average each recording's window softmax scores exactly as the deployed inference code does. Additional patient-level metrics average recording scores per subject. Validation patient macro F1 selects the checkpoint; validation recording negative log-likelihood breaks ties. The test set is evaluated only after selecting the checkpoint.

These are limited same-source prototype results, not clinical validation. A single pneumonia participant in the validation and test groups is insufficient to establish general disease recognition. Electronic-stethoscope recordings differ from free-air phone/browser/INMP441 recordings; microphone, chest placement, background sounds and population changes can affect predictions. Arbitrary sounds, cough-only clips and untrained conditions are not validated inputs. Scores are not calibrated disease probabilities. Do not use the model for medical decisions.

## Run or reproduce

From the project root:

```powershell
.\.venv\Scripts\python.exe backend/train_audio.py --epochs 60
```

A completed checkpoint is installed atomically as `backend/models/audio.pt`. Model loading uses the same architecture/preprocessing identifier as the current backend, so an already-running compatible server detects the new file on its next health/inference request. No synthetic patient samples, filename-based disease lookup or copied asthma examples are used.

Outputs:

- `backend/models/audio.pt`: actual trained weights, labels, preprocessing version, unsupported-class and evaluation notes.
- `backend/models/audio_metrics.json`: measured training history, recording/patient test metrics and evaluation protocol.
- `backend/models/audio_run/preparation.json`: split counts and excluded classes.
- `backend/datasets/audio_manifest.csv`: real patient-disjoint training/validation/test assignments.

Use Upload WAV / FLAC, Record Audio or Load ESP32 Audio, then Analyze Audio. The dashboard lists supported classes and explicitly reports that Asthma is unsupported. The ESP32 connection status is independent of whether uploaded audio can be analyzed.

## Measured installed-model results

The selected checkpoint was epoch 51. Held-out recording accuracy was 20/35 (57.1%), balanced accuracy 54.6% and macro F1 0.518. Patient-aggregate accuracy was 13/19 (68.4%), balanced accuracy 72.3% and macro F1 0.556. The test set contains just one pneumonia participant. Normal-breathing recording recall was only 2/5 (40.0%). These results show limited and unreliable disease recognition. This model enables demonstration of the recording/analysis workflow; it is not suitable for health screening decisions. Full confusion counts and history are retained in `backend/models/audio_metrics.json`.
