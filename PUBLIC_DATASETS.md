# Public respiratory dataset samples

This project now has an acquisition script for real, labelled chest X-rays and respiratory audio, including source-labelled normal/healthy controls. These samples are for dataset exploration and prototype development. They do not supply a trained model or clinical accuracy values.

## Downloaded starter set

| Modality and source label | Downloaded files |
|---|---:|
| audio: Asthma | 1 |
| audio: COPD | 20 |
| audio: Normal Breathing | 20 |
| audio: Pneumonia-related respiratory pattern | 20 |
| xray: Normal | 40 |
| xray: Pneumonia | 20 |
| xray: Tuberculosis | 20 |

All 141 selected media files decoded successfully and matched their saved SHA-256 values. No exact byte duplicates were found within this selected set. These are software/data-integrity checks, not medical validation.

## Download and verify

From the project root:

```powershell
python backend/fetch_datasets.py --per-class 20
.\.venv\Scripts\python.exe backend/inspect_datasets.py
```

The second command uses NumPy and Pillow from the existing project environment. If your environment is elsewhere, use its Python interpreter.

The default is a reproducible starter subset of up to 20 files per class per source, selected across available patients for audio. It is not the complete training dataset. To request all supported source files, run the first command with `--per-class 0`; allow several GB of storage and download time. Files already present are reused. Changing subset size rebuilds the inventory of selected files; older downloaded files remain on disk.

## Sources and classes

| Source | Modality | Included labels | Source |
|---|---|---|---|
| Chest X-Ray Images (Pneumonia), version 2 | X-ray | Normal, Pneumonia | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| Tuberculosis (TB) Chest X-ray Database, version 3 | X-ray | Normal, Tuberculosis | https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset |
| Respiratory Sound Database / ICBHI, version 2 mirror | WAV audio | Normal Breathing, Asthma, COPD, Pneumonia-related respiratory pattern | https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database |

The ICBHI source diagnosis `Healthy` maps to `Normal Breathing`; `Pneumonia` maps to the project's `Pneumonia-related respiratory pattern`. The labels describe source participants, not a diagnosis inferred by this script. Acoustic annotations such as crackles/wheezes are not relabelled as diseases. Other source diagnoses are excluded from this quoted-scope subset. Duplicate directory copies in Kaggle archives are collapsed by class and filename when selecting files.

The TB data card describes 3,500 normal and 700 publicly accessible TB images; additional restricted TB cases require an agreement and are not downloaded here. Counts of selected local files and distinct audio patients are computed in `coverage.json`. Use those measured local counts rather than assuming all published records were downloaded.

## Files and measured values

All data is local under `backend/data/public_datasets/`, already covered by the project's `backend/data/` Git ignore rule:

- `xray/<label>/<source>/`: original PNG/JPEG source images, including Normal.
- `audio/<label>/icbhi/`: original WAV source recordings, including Normal Breathing.
- `inventory.csv`: paths relative to backend, source file paths, source and project labels, source split where supplied, verified audio patient ID, byte size and SHA-256.
- `file_values.csv`: decoded image size and grayscale pixel statistics; audio sampling rate, channel count, bit depth, duration, peak and RMS amplitude. Blank fields mean not applicable. These values are measured from files, not generated confidence or severity scores.
- `coverage.json`: selected versus indexed unique-filename counts and audio patient counts.
- `validation.json`: decode/hash checks, exact duplicates and measured local class counts.
- `metadata/`: cached source indices, source metadata/descriptions and the ICBHI patient diagnosis table.

Audio peak/RMS values are digital full-scale amplitudes, not calibrated decibels or sound pressure. Image statistics are 8-bit grayscale values, not Hounsfield units or disease severity. The scripts do not resample or transform the original media files.

## Training limitations

ICBHI has only one asthma participant/recording in this release. That supports an example, but cannot support patient-independent asthma train/validation/test partitions. Do not duplicate that patient into multiple splits or report a validated four-class model from these samples.

An additional source with specialist asthma/COPD annotations is [Chest-wall lung sounds, Mendeley Data v3](https://data.mendeley.com/datasets/jwyy9np4gv/3). Its WAV files and combined annotation file require separate acquisition/inspection; they are not included by this downloader. The source page specifies CC BY 4.0. Its acoustic `N` annotation is a normal sound and must not by itself be interpreted as a healthy patient diagnosis.

X-ray source filenames do not establish verified patient identities. Their inventory patient IDs are deliberately blank. Consult source metadata to establish grouping before producing patient-disjoint train/validation/test manifests. The existing training manifests remain unchanged. The original pneumonia split is retained only as provenance, not assumed free from leakage.

The X-ray datasets and ICBHI contain different cohorts; they are not paired audio/X-ray tests of the same people. Do not combine arbitrary sample pairs as corroborating patient evidence. Pediatric pneumonia images and the TB collection can also differ in age, acquisition and institution; mixing them into one classifier requires checking source confounding and independent validation.

Kaggle metadata returned nonstandard/unknown license labels for these mirrors. Original descriptions and metadata are retained for review; public download availability does not imply unrestricted commercial or redistribution permission. Retain source attribution and confirm terms for your intended use. No source usage agreements are accepted by the scripts.
