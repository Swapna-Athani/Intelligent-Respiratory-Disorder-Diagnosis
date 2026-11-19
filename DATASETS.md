# Real dataset preparation

Real public dataset starter files, including Normal/Healthy controls, are acquired using `backend/fetch_datasets.py`. See [PUBLIC_DATASETS.md](PUBLIC_DATASETS.md) for source links, download commands, measured audio/image values and coverage limits. These starter files are not automatically assigned training splits or used to generate application results. No fake results or pretrained weights are bundled. Training requires curated real labelled files that you are permitted to use. Keep patient identifiers de-identified.

## Audio manifest

`backend/datasets/audio_manifest.csv` starts with only:

```csv
path,label,patient_id,split
```

Fill with your actual file paths, labels and patient IDs. Paths are relative to the manifest file (absolute paths also work). `split` must be train, val or test.

Use these exact labels for the full quoted scope:

- Normal Breathing
- Asthma
- COPD
- Pneumonia-related respiratory pattern

Pass `--normal-label "Normal Breathing"`. `Normal` is also accepted if that is how your actual normal label is defined. Do not map an acoustic label such as crackles/wheeze directly to a disease unless the source provides the corresponding disease annotation. Do not invent missing classes or manufacture patient IDs that separate files from the same patient.

Accepted inputs: PCM WAV / FLAC, mono or stereo, 1–120 seconds. The pipeline resamples to 16 kHz, averages channels, applies a 100–2000 Hz band-pass and simple spectral noise-floor subtraction, normalizes and derives 64-band log-Mel and 20 MFCC features. The noise-floor subtraction is a simple signal-processing baseline and can affect quiet sounds; assess it on your actual recordings.

Training samples a random 5-second window per file each epoch, padding shorter recordings. Validation/test metrics use the first 5 seconds per file. Inference splits the supplied recording into 5-second windows (last windows shorter than one second are omitted), computes each window's softmax, then averages scores. Therefore the recorded held-out metrics are segment metrics, not a separately validated recording-level score. Validate the exact deployment aggregation on independent patient recordings before relying on it.

## X-ray manifest

Same columns in `backend/datasets/xray_manifest.csv`. For a dataset supporting the stated scope, use `Normal`, `Pneumonia`, `Tuberculosis`. Additional abnormal labels can be used if they are supported by the chosen real dataset; the model learns only provided classes. The project uses single-label classification, not multi-label coexisting disease detection.

PNG/JPEG, at least 64×64, up to 25 megapixels. The shared pipeline performs EXIF orientation correction, grayscale conversion, autocontrast, median filtering, resizing to 224×224, and normalization to [-1,1]. It does not perform lung segmentation or validate X-ray view/quality. DICOM and Grad-CAM are not included. Grad-CAM was optional in the scope.

## Splits and evaluation

Assign patients to training/validation/test sets before generating manifests. Keep all visits and paired modalities for a patient in the same split. The script enforces splits within one manifest; cross-modality patient linkage must be checked during dataset curation. Every class must occur in every split. Avoid comparing models using overlapping patients, institution-specific artifacts, duplicate or near-duplicate scans, or mislabeled disease classes.

The script verifies exact byte duplicates across splits and patient overlap, but cannot detect all near-duplicates or wrong patient identifiers. Keep provenance, source license, split counts, and annotation rules with your dataset. Model scores are not calibrated clinical confidence and do not establish a medical diagnosis.

## Training commands

Run in backend with its virtual environment active:

```sh
python train.py --kind audio --manifest datasets/audio_manifest.csv --normal-label "Normal Breathing" --epochs 20
python train.py --kind xray --manifest datasets/xray_manifest.csv --normal-label Normal --epochs 20
```

Training rejects header-only templates until real records are supplied. Change epochs and batch size according to actual data and CPU memory. Performance is determined by the dataset and validation; no fixed accuracy is claimed.
