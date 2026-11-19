# Quotation scope audit

Audit date: 27 October 2026 (Asia/Calcutta).
Project reviewed: `C:\Users\USER\Downloads\ks5785\ks5785`.
Quotation reviewed: the supplied KS5799 / SPN-26092026-KS5799 text for AI-Based Respiratory Disorder Detection Using Lung Sound and Chest X-Ray Analysis with Web Dashboard.

## Assessment

The software prototype implements most dashboard, upload, inference, visualization, comparison, history and report functions. It does **not** yet demonstrate the complete quoted embedded/hardware system. The required Asthma audio class is unsupported, and the installed audio model has limited recognition performance. This should not be described as complete delivery of every quoted feature.

No completion percentage is assigned: working buttons, implemented code, measured model accuracy and physically verified hardware are different kinds of evidence.

## Requirement-by-requirement comparison

| Quoted section / requirement | Status | Evidence and remaining work |
|---|---|---|
| A. ESP32 + INMP441 + OLED + LEDs + battery + wireless acquisition | Not established in this repository | No ESP32 firmware, I2S driver, pin map/wiring diagram or hardware acceptance evidence was found. Physical hardware may exist separately; it was not inspected. Firmware must acquire approximately 16 kHz/16-bit audio and transmit real samples/telemetry. |
| B. Real-time acquisition, buffering, segmentation and wireless transmission | Partial integration only | `loadDeviceAudio()` imports a complete WAV from Firebase when clicked. Device heartbeat/status is read in real time. No firmware-driven capture/streaming loop or automatic continuous audio processing exists in the reviewed files. |
| C. Audio noise reduction, filtering, normalization, segmentation, MFCC and Mel-spectrogram | Implemented | Shared functions in `backend/processing.py`, `backend/train_audio.py` and `backend/inference.py`; engineering preprocessing and real-audio upload tests pass. Noise reduction is a simple spectral-floor baseline, not demonstrated to handle all hardware/noise environments. |
| D. CNN classification: Normal, Asthma, COPD, pneumonia-related patterns | Partial; required class missing | Installed audio labels are Normal Breathing, COPD and Pneumonia-related respiratory pattern. Asthma is excluded because the source contains only one participant. Held-out recording accuracy is 20/35 (57.1%) from 19 subjects. Obtain additional labelled asthma participants and improve independent audio/device evaluation. |
| E. X-ray selection, upload, preview, validation, processing and results | Implemented for PNG/JPEG | Frontend selection/decode checks, pending test association, API processing and result display exist; real-model upload tests pass. Validation checks raster format, size/dimensions and decoding, not whether an arbitrary photograph is a suitable chest X-ray. |
| F. X-ray preprocessing | Implemented with a different active recipe | The installed ResNet18 model uses RGB conversion plus pretrained resize/crop/normalization. Autocontrast/median filtering exists for the legacy CNN, but is not used by the installed transfer model. The quotation says the listed enhancement/noise steps can be included, so their absence from the active recipe is not treated as a mandatory missing feature. |
| G. Normal/Pneumonia/Tuberculosis X-ray classifier and scores/status | Implemented as an exploratory model | All three labels load in the live backend. Same-split test accuracy is 52/60 (86.7%); fresh same-source image audit is 58/60 (96.7%). Patient independence and clinical performance are not established. Extra abnormalities depend on the dataset and are not currently supported. |
| H1. Dashboard system/model/device status and normal/abnormal/test counts | Implemented | `frontend/src/main.jsx` computes counts from saved analyzed tests and marks device connection from heartbeat freshness. Device status has not been verified with physical hardware. Counts are display summaries, not diagnostic statistics. |
| H2. Recording, waveform, Mel-spectrogram, audio prediction/percentage | Implemented for completed recordings | WAV/FLAC selection, browser microphone recording and manual ESP32 WAV import are supported. Waveform/spectrogram appears after recording stops or a file is loaded; there is no continuously updating waveform during capture. |
| H3. X-ray dashboard | Implemented | Preview, independent result, model score and normal/abnormal display are present. Filename/folder-independence checks pass. |
| H4 / I. Independent combined analysis and disagreement handling | Implemented | `combine()` displays both individual outputs and scores, recognizes similar pneumonia-related outputs and shows disagreement without forcing a single diagnosis. This matches the quotation's independent comparison, rather than requiring a fused model. |
| H5. Previous tests, sample ID, dates and both results/scores | Implemented; live-account acceptance still needed | Firebase listeners/history table and per-user paths exist. Backend integration tests mock authentication/storage; this audit did not independently perform a fresh signed-in browser save/read cycle. Existing user screenshots show saved-test counts, but that does not verify every history/error path. |
| H6. Report with sample information, modality results, scores and dates | Implemented; live-account export acceptance still needed | PDF endpoint includes sample ID, test ID, timestamps, predictions, normal/abnormal status, model scores and model versions. PDF generation passes an engineering test with mocked Firebase data. Patient details currently mean a sample identifier, not a full demographic/clinical chart. |
| J. Physical OLED ready/recording/processing/prediction/confidence/status | Partial UI support; physical function unverified | The dashboard renders a software OLED panel and can display supplied device text. No physical OLED firmware or read-back/result display implementation was found. A web panel alone does not satisfy verification of a physical OLED. |
| Registration, login and profile | Implemented additions | Firebase email/password account screens and display-name profile editing are present. These are useful additions but do not close the embedded or classification gaps. |
| Deployment and support handover | Not verified | Local backend is running and a production frontend build succeeds. A hosted deployment, physical delivery and the promised support arrangements cannot be verified from repository/source checks. |

## Optional or conditional items

- Grad-CAM/heatmaps: not implemented. The quotation explicitly calls this optional.
- Additional X-ray disease classes: not implemented beyond Normal/Pneumonia/Tuberculosis; the quotation makes these dataset-dependent.
- DICOM: not supported. The quotation does not require this file format, so it is not treated as a missing contractual scope item.
- Clinical validation: not established. The quotation describes a preliminary prototype, so clinical deployment was not assumed to be promised. The current model scores must still not be described as calibrated disease probabilities.

## Verification performed

- Read the supplied quotation, frontend application/auth/profile source, backend API/preprocessing/inference/source-data preparation and device data contract.
- Live `/api/health` confirms the current compatible server can load **both** audio and X-ray models with their matching preprocessing versions.
- Backend test discovery: 16 tests passed, including actual audio/image model inference, upload endpoints, filename independence, patient split checks, model compatibility and PDF generation.
- Frontend upload/sample-ID checks: 4 tests passed.
- Production frontend build: passed. Vite emitted a bundle-size warning; it did not prevent the build.
- Real-model API engineering tests mock Firebase authentication and persistence and do not write patient records. Passing them does not establish live-cloud rules, a browser microphone/hardware session or medical accuracy.
- Physical ESP32/INMP441/OLED/LED/battery operation and a new signed-in full browser workflow were not available for direct verification in this audit.

## Priority work before claiming all quoted features are complete

1. Obtain and test the ESP32/INMP441 firmware, wiring, power, status LEDs, wireless audio transfer and physical OLED result display. Validate actual 16 kHz/16-bit recordings and communication with this backend.
2. Add genuine multi-patient asthma data and validate the required four-class audio scope without putting one person's recordings into multiple splits. Improve the current weak audio model before presenting it as a dependable screener.
3. Finish automatic continuous device capture/transfer/analysis where required by the real-time embedded scope. Verify failure/disconnection recovery and timing with the actual device.
4. Perform signed-in acceptance tests for capture/upload -> save -> reload history -> combined results -> PDF export, then verify on the deployment machine and physical OLED. Keep both modalities associated with the correct sample/test.
5. Reconcile documentation: the quotation reference is KS5799 while the project is ks5785; README also retains older statements about no pretrained model/header-only manifests that are no longer accurate for the installed checkpoints/audio manifest. This audit did not rename the project or change the quoted scope.

The quotation does not specify a minimum numerical accuracy target. The low audio accuracy is therefore reported as a substantive quality limitation, not as a claimed violation of an unstated target. The missing Asthma class and unverified embedded/OLED deliverables remain clear scope gaps.
