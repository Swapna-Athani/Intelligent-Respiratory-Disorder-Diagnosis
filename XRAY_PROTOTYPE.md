# Installed X-ray model and evaluation

The application now uses a frozen ImageNet-pretrained ResNet18 image feature extractor and a classification head trained on the project's real Normal, Pneumonia and Tuberculosis images. The previous small CNN checkpoint is preserved under `backend/models/backups/`.

## Measured results

| Evaluation | Previous small CNN | Installed transfer model |
|---|---:|---:|
| Same 60-image test set: accuracy | 37/60 (61.7%) | 52/60 (86.7%) |
| Same test set: macro F1 | 0.545 | 0.859 |
| Same test set: TB recall | 3/15 (20.0%) | 14/15 (93.3%) |
| Fresh 60-image audit: accuracy | Not evaluated | 58/60 (96.7%) |

The fresh audit contains 20 Normal, 20 Pneumonia and 20 Tuberculosis images from the same public sources. Correct classifications were 20/20 Normal, 18/20 Pneumonia and 20/20 Tuberculosis. Exact byte duplicates found under other filenames were excluded and replaced before any audit evaluation; exclusions are retained in `backend/data/public_datasets/fresh_xray_audit/excluded_duplicates.json`.

These are small **image-level** evaluations. Source metadata does not establish patient identities. Patient overlap, near-duplicate scans and age/institution/source confounding cannot be ruled out. The audit is not an external hospital study, and its score is not a guarantee for arbitrary uploaded images. The original test set was previously inspected during baseline development; the fresh audit provides an additional untouched image check, but still does not establish patient-independent or clinical accuracy. The model remains an exploratory prototype and must not be used for medical decisions.

## What changed

- Replaced training a small CNN from scratch with pretrained visual features and a supervised three-class classifier.
- Preserved the original 280 training / 60 validation / 60 test image assignments.
- Fitted feature normalization on training images only. The pretrained backbone remains frozen.
- Selected the classifier using validation macro F1, with validation loss breaking ties. Test/audit scores did not tune or select the classifier.
- Evaluated a fresh, exact-byte-disjoint image audit after classifier selection.
- Preserved the strict patient-disjoint `train.py` workflow and its leakage checks.
- Installed a checkpoint only after validation macro F1 improved over the previous model, with the previous weights and metrics backed up.

The feature extractor comes from [official TorchVision ResNet18 ImageNet weights](https://docs.pytorch.org/vision/2.0/models/generated/torchvision.models.resnet18.html). ImageNet pretraining is not a claim of medical validation. RGB resize/crop/normalization follows those weights' documented transforms; the trained checkpoint records the architecture and matching preprocessing version. Backend inference makes no network download: all required model state is bundled in the installed checkpoint.

## Folder and filename behavior

Predictions use decoded image pixels. Source folders supply supervised labels only during dataset preparation/training. During an upload the folder name and file name are not classifier features. Integration checks copy identical bytes into folders named `Normal` and `Tuberculosis`, and upload them under opposite filenames; both paths must return identical scores.

A correct folder label does not guarantee a correct prediction, and incorrect predictions remain possible, including on images from these datasets. The model supports only Normal, Pneumonia and Tuberculosis; it is not a general detector for all diseases or for arbitrary non-X-ray photographs.

## Start the updated backend

The new architecture requires restarting the already-running backend. Press Ctrl+C in that backend terminal, then run from the project root:

```powershell
.\.venv\Scripts\python.exe backend/app.py
```

Refresh the frontend and select an image again. The audio model is separate and is unchanged.

## Reproduce and inspect

With the project's environment and downloaded files present:

```powershell
.\.venv\Scripts\python.exe backend/fetch_xray_audit.py
.\.venv\Scripts\python.exe backend/train_xray_transfer.py
.\.venv\Scripts\python.exe -m unittest discover -s backend -p "test_*.py" -v
```

The current environment uses torch 2.14.1+cpu and torchvision 0.29.1+cpu, installed from the official PyTorch CPU wheel index. Install compatible torch/torchvision builds together on other machines. The first training run downloads official pretrained weights into `backend/data/pretrained/`; later feature extraction can reuse cached features. Rerunning the same candidate does not replace the current model unless validation improves. Do not tune to the published audit results; acquire a new independent audit for future model changes.

Outputs:

- `backend/models/xray.pt`: installed transfer model, three labels, preprocessing version and evaluation limitations.
- `backend/models/xray_metrics.json`: measured validation/test/audit results, training history and selection rules.
- `backend/models/xray_transfer_run/`: candidate weights, feature cache and complete run metrics.
- `backend/models/xray_prototype_run/image_split_manifest.csv`: unchanged image split assignments and source/checksum records.
- `backend/data/public_datasets/fresh_xray_audit/manifest.csv`: fresh audit source labels and checksums.

The old `train_xray_prototype.py` command trains the weaker baseline and can overwrite the installed model. Use `train_xray_transfer.py` for this upgraded workflow. No accuracy target is guaranteed.
