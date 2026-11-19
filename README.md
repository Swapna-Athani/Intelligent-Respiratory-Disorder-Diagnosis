# Respiratory Screening Dashboard

Application folder/package: **ks5785**.

React frontend + Python Flask/PyTorch backend + your Firebase client configuration. The layout follows the supplied image: navy sidebar, summary cards, patient/sample information, respiratory waveform, Mel-spectrogram, X-ray preview, independent predictions, combined comparison, history, reports and OLED-style device panel. No quotation number appears in the interface.

**Real public dataset starter samples can be downloaded with `backend/fetch_datasets.py`; see [PUBLIC_DATASETS.md](PUBLIC_DATASETS.md) for healthy/disease coverage and measured values. There is no demo application data, fake device connection, fabricated prediction, or pretrained model in this package.** Counts start at zero. Model scores appear only after a trained model processes a real input. Medical screening models here are research baselines; they are not clinically validated.

## 1. Prerequisites

- Node.js 20.19+ or 22.12+.
- Python 3.10.11, with pip and venv. Create the environment on the
  machine where it will be used; virtual environments are not portable between
  computers.
- A Firebase project with Email/Password authentication enabled. Use the Register screen to create an account, or Log in with an existing account. Credentials are entered at run time and are not embedded in this package.
- Firebase Email/Password authentication must already be enabled, and Realtime Database rules must allow the signed-in account's own project path.

No Firebase Admin SDK, service account, private key, or admin login is used. A Firebase web API key identifies the project; it does not replace user authentication or grant database permission.

## 2. Start the Python backend

Open a terminal in `ks5785/backend`. Create the environment there and run the
server from the same directory.

Windows PowerShell:

```powershell
# Run these commands from ks5785/backend.
deactivate 2>$null
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python app.py
```

The required output is `Python 3.10.11`, and the executable path must be
inside `backend\.venv`. The repository includes `backend\.python-version` as
the project version marker. If a terminal still reports a Python 3.13 path,
close that terminal, open a new PowerShell window, and activate the environment
again before running the commands above. Do not copy a `.venv` from another
machine.

Linux/macOS:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python app.py
```

The CPU build is sufficient for this implementation. Backend starts at `http://127.0.0.1:5000`, using Waitress. No training is performed automatically when starting the server. The terminal prints a loading message followed by the backend address and model availability, then stays occupied while serving requests. Leave it open; this is expected. Visit `http://127.0.0.1:5000/api/health` to check it. The dashboard is served by the React frontend at `http://localhost:5173`, not by opening the backend root address. Download progress and `training_ready: false` do not indicate a server failure: downloaded data still requires curated training splits and trained model weights.

## 3. Start the React frontend

Open a second terminal in `ks5785/frontend`:

```sh
npm install
npm run dev
```

Open the local URL, normally `http://localhost:5173`. The development proxy forwards `/api` to the Python backend. The application opens on the **Log in** screen. Choose **Register** to create an email/password account, or log in with an existing account. Registration requires a password of at least 8 characters and matching confirmation. Successful authentication opens the dashboard; signing out returns to Log in. Authentication persists for the browser session.

If npm reports `EBUSY` for `esbuild` or `EPERM` while cleaning `node_modules` on
Windows, stop any running Vite/Node terminal and pause OneDrive synchronization
for this folder. From `ks5785/frontend`, run:

```powershell
Remove-Item -Recurse -Force node_modules
npm ci
npm run dev
```

`npm ci` uses the committed `package-lock.json`. Do not delete the lockfile or
run the install while another `npm`, `vite`, or frontend preview process is
using this directory. If OneDrive continues to lock files, move or clone the
project to a local non-synchronized folder before installing dependencies.

For a production build:

```sh
npm run build
npm run preview
```

The included `frontend/dist` is prebuilt. Serve it through an HTTP host, not by double-clicking index.html. Production hosting needs a reverse proxy for `/api`, or a `.env` containing `VITE_API_BASE=https://your-backend` followed by a rebuild. Set the Python `ALLOWED_ORIGINS` environment variable to that frontend origin for cross-origin hosting. Do not expose the local backend over plain HTTP on the public internet.

## 4. Firebase project folder and rules

Client configuration is in `frontend/src/firebase.js`; the Python client REST settings are in `backend/config.py`. Your provided project values are already inserted. Measurement ID and storage bucket are retained in configuration; Analytics and Storage are not required for the included workflows.

The application initializes this data location after successful sign-in:

```text
ks5785/users/<signed-in-uid>/metadata
ks5785/users/<signed-in-uid>/tests/<test-uuid>
ks5785/users/<signed-in-uid>/device
ks5785/users/<signed-in-uid>/incomingAudio
```

Firebase creates parent nodes when the first real write occurs. Empty test/device/audio nodes are not seeded with dummy data. The folder on disk is also named `ks5785`.

`firebase.database.rules.json` contains a rules fragment for authenticated users to read/write only their own user subtree. Merge the `ks5785` branch into your existing rules; do not replace unrelated project rules. Broad allow rules at an ancestor (such as `.read: true` at the database root) override child restrictions and must be resolved by the project owner before storing patient information. The app does not alter deployed database rules.

The frontend reads records through Firebase's real-time SDK. Python verifies the supplied Firebase ID token with the client authentication REST API, then reads/writes Realtime Database using **that same user token**. It cannot bypass database rules. The project web configuration alone is insufficient to test your live account from this package.

## 5. Train on real labelled datasets

The three-class audio prototype is prepared/trained with `python backend/train_audio.py --epochs 60`. See [AUDIO_MODEL.md](AUDIO_MODEL.md) for patient-disjoint splits, recording-level evaluation and the unsupported Asthma class.

The installed exploratory X-ray model uses pretrained ResNet18 features. Reproduce it with `python backend/train_xray_transfer.py` from the project root, after `python backend/fetch_xray_audit.py`. Restart the backend after this architecture upgrade. See [XRAY_PROTOTYPE.md](XRAY_PROTOTYPE.md). This separate workflow uses image-level splits and does **not** establish patient-independent or clinical accuracy. The patient-disjoint training workflow below remains unchanged.

See `DATASETS.md`. Training uses a manifest with `path,label,patient_id,split`. Header-only manifest templates are included, with no sample records.

Populate both manifests using actual, licensed, de-identified dataset files and clinically sourced labels. Ensure each patient's files are confined to exactly one split, and every class is represented in train, val and test. The training script rejects patient overlap, duplicate file paths, and identical file bytes occurring across splits.

From `backend`:

```sh
python train.py --kind audio --manifest datasets/audio_manifest.csv --normal-label "Normal Breathing" --epochs 20 --batch-size 8
python train.py --kind xray --manifest datasets/xray_manifest.csv --normal-label Normal --epochs 20 --batch-size 8
```

Outputs:

- `models/audio.pt` and `models/xray.pt` — trained weights, labels, preprocessing version, model version and validation metrics.
- `models/audio_metrics.json` and `models/xray_metrics.json` — training history, held-out accuracy, balanced accuracy, macro F1 and confusion matrix.

The best validation macro-F1 checkpoint is selected, then evaluated on the held-out test set. No accuracy target is promised. Do not interpret passing software tests as model or clinical validation. A model only supports classes present in its real training manifest.

The included compact CNN trains from scratch on CPU. Audio input has two channels: normalized log-Mel and resized MFCC feature maps. Image input is grayscale 224×224. This is an executable baseline training workflow; dataset curation and independent clinical validation remain necessary.

## 6. Use the dashboard

1. Log in or register. All history and counts load from your own Firebase path. Open **Profile** in the sidebar or your account name in the header to update your full name and review account details.
2. Enter a patient/sample ID. Prefer a de-identified sample identifier.
3. Upload WAV/FLAC, record audio through the browser microphone, or load an ESP32 WAV recording from Firebase. Browser microphone access requires localhost/HTTPS and your permission. Recording stops after 60 seconds.
4. The backend prepares the actual waveform and spectrogram. Click **Analyze Audio** after training the audio model.
5. Enter a patient/sample ID (for example `1`), then select a real PNG/JPEG chest X-ray. The application validates the image and saves a pending test linked to that ID in Firebase; the original image remains a local selection and is not archived. Click **Analyze X-Ray** after starting the backend and training the image model. The application checks backend/model readiness before analysis. File validation is not a validated detector of whether an image is a suitable chest X-ray.
6. Both results attach to the same test ID and are saved through Firebase. Start **New Test** before working on a different patient/sample. New Test keeps the entered sample ID so it is not accidentally lost; edit it for a different patient before uploading.
7. Combined Analysis compares the independent results without averaging scores or generating a single fused diagnosis.
8. Select a saved test in history to review it. Generate a PDF from the Reports section or current test.

A missing model produces an explicit “not trained” error, with no prediction. Uploads must be below 25 MB. Audio must be 1–120 seconds. Image dimensions must be at least 64×64 and no more than 25 megapixels; DICOM is not supported. Scores are uncalibrated softmax outputs, not probabilities of having a disease.

Test counts include saved tests with at least one analyzed modality. A test is counted as abnormal if either available modality flags abnormal; otherwise it is counted as normal. Unanalyzed pending records do not count. This is a display aggregation, not a diagnostic decision. History retains each individual result and score.

Input files are processed in memory. Raw original files are not archived. Downsampled waveform, spectrogram, X-ray thumbnail and prediction metadata are stored in Firebase with each result, so history can display them. Treat these as sensitive health data and apply appropriate access and retention policies. No browser localStorage dataset is used.

## 7. ESP32 / Firebase data contract

Your firmware or gateway should authenticate as the same Firebase user (or an account with deliberately granted rules) using the normal Firebase client sign-in and refresh flow. Never put an Admin key in firmware.

For the device card, write real telemetry to:

```text
/ks5785/users/<uid>/device
```

Expected fields:

- `lastSeen`: Unix time in **milliseconds**, updated approximately every 5 seconds with a synchronized clock.
- `status`: actual device state, e.g. recording or ready.
- `oled`: optional multiline string from the actual device display.

The UI marks a device disconnected after 15 seconds without a valid recent heartbeat; it never assumes connected.

For actual recorded audio, write a WAV clip to:

```text
/ks5785/users/<uid>/incomingAudio
```

Required field: `wavBase64`, containing the base64 of a complete valid PCM WAV file, without a data-URL prefix. An `id` or recording timestamp may also be stored. A 5-second, mono, 16-bit, 16 kHz clip is about 160 KB before base64 and fits this route. Keep each clip under 1 MB. The clip must include a WAV header, not just raw I²S bytes. Click **Load ESP32 Audio**, then **Analyze Audio**. This imports actual Firebase audio; it does not simulate a sensor. No ESP32 firmware is included in this web/backend deliverable.

Optional firmware reads the resulting `tests/<test-id>/audio` record to update its physical OLED. The dashboard's OLED-style panel shows device text when supplied, otherwise the selected saved audio result, clearly labelled.

## 8. API

All endpoints except health require `Authorization: Bearer <Firebase ID token>`.

- `GET /api/health` — backend/model-file availability, no patient data.
- `POST /api/preview/audio` — multipart `file`, actual waveform/Mel preview.
- `POST /api/preview/xray` — multipart `file`, validated raster preview.
- `POST /api/analyze/audio` or `/xray` — multipart `file` and owned `testId`; trained inference and Firebase save.
- `GET /api/report/<test-id>` — PDF generated from owned Firebase records.

Training is a local command-line workflow, not an exposed web endpoint. Do not load model checkpoints from untrusted sources.

## 9. Verification

```sh
# From backend, with the Python environment active:
python -m unittest test_backend.py -v
# From frontend:
npm run build
```

Engineering tests use ephemeral synthetic signal/image fixtures inside memory or temporary directories solely to verify software behavior. They never populate your application/Firebase or ship a trained model/dataset. They do not evaluate medical accuracy.

## Included files

- `frontend/` — React/Vite source, client Firebase config, lockfile, compiled dist.
- `backend/` — Python API, preprocessing, CNN, training CLI, engineering tests, requirements.
- `firebase.database.rules.json` — scoped example rule branch.
- `DATASETS.md` — training preparation and label instructions.

### Backend/model compatibility

The health endpoint now verifies that model weights load with a supported architecture and matching preprocessing; file presence alone is not readiness. It returns `backendVersion`, boolean model readiness and detailed compatibility errors. The frontend detects older backend instances before analysis. Model objects are cached by checkpoint path, modification time and size, so an atomically replaced checkpoint is loaded on the next request.

If multiple old project backend instances are running, stop them before starting another copy. A background project backend may have logs at `backend/data/backend_stdout.log` and `backend/data/backend_stderr.log`, with its launcher record in `backend/data/backend_process.json`. Do not start a second copy while the verified server is listening on port 5000.
