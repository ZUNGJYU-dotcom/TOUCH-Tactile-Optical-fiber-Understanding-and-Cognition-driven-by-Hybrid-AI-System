# TOUCH

Research software for optical-fiber tactile sensing. The current Stable Operator
runtime uses a trained ordinary-FBG 512-point BaySpec/Sense High Sensitivity /
300 us joint nine-FBG model to estimate:

- contact or no contact;
- approximate manual contact position `P11`-`P33`;
- continuous optical `Fz` with no fixed software display/output ceiling; the
  current model was trained with approximately 0-6.98 N supervision.

The model consumes optical features only at runtime. PX6D `Fz` was used as
synchronized training and validation supervision and is not required for
optical-force inference. The retained PD-voltage and earlier optical-intensity
applications remain separate.

## mFBG Expansion Contract

The retained ordinary-FBG model remains the active runtime. A separate
`mfbg_intensity_3x3` profile has now been added as the future primary sensor
path. It provides nine-channel spectral-window intensity demodulation,
multi-region frame structures, baseline-aware recording adapters, and isolated
`/api/mfbg-intensity/*` endpoints without changing the existing ordinary-FBG
inference contract.

Real mFBG 3x3 mode is still disabled. The 1540-1580 nm channel table is a
preliminary target plan, measured wavelengths are not populated, no new mFBG
training data have been used, and the continuous surface remains a raw coupled
optical-response proxy rather than calibrated force or pressure. See
[docs/mfbg_intensity_profile.md](docs/mfbg_intensity_profile.md) for the
demodulation, frame, recording, and activation contracts.

## Latest Update - v0.19.25 Stable

Stable v0.19.25 promotes the user-tested v0.19.25 Beta runtime without changing
the deployed model artifact. Contact uses LightGBM over the complete
baseline-relative spectrum plus the joint nine-FBG fingerprint, position uses
a random forest over the joint fingerprint, and optical `Fz` uses histogram
gradient boosting. Runtime inference remains optical-only; PX6D is supervision
and diagnostics, never an inference input.

The model contains 49,853 same-day High Sensitivity / 300 us frames from 161
complete sessions. Blind2 is excluded. The formerly blind Blind1, Blind3, and
Blind4 batches were used during model development and are therefore not
claimed as independent blind evidence.

The exact runtime replay produced zero activations in 10,589 dedicated-idle
frames, 99.85% active-contact coverage, and no emitted wrong-position frame.
Every emitted position was correct in this answer-known replay; 44 uncertain
frames were withheld during confirmation. Grouped out-of-file evaluation
reported 98.06% raw position accuracy and 0.310 N force MAE.

The release-latch repair ignores late concurrent SDK frames instead of
rewinding temporal state. If the contact classifier remains saturated after
release, contact can also clear only after the complete nine-FBG spectrum has
recovered at least 60% from an armed contact peak and remained quiet with low
spatial confidence for two physical frames spanning at least 180 ms. The
trained clean-session baseline remains fixed.

Stable has an isolated copy of the validated contact-state configuration and a
separate High Sensitivity / 300 us user-settings file. Old Stable preferences
and future Beta experiments cannot silently change its acquisition behavior.
The promotion capture ended released after 54 live frames and then produced
zero false activation across 31 additional idle frames.

See [CHANGELOG.md](CHANGELOG.md) for the complete release summary.

## Download Stable v0.19.25

Download the single-file Windows x64 application from the
[v0.19.25 GitHub release](https://github.com/ZUNGJYU-dotcom/TOUCH-Tactile-Optical-fiber-Understanding-and-Cognition-driven-by-Hybrid-AI-System/releases/tag/v0.19.25):

```text
TOUCH-Stable-v0.19.25-Windows-x64.exe
```

Published artifact:

| Property | Value |
| --- | --- |
| Size | 103,285,538 bytes (98.50 MiB) |
| SHA-256 | `7963470D59CDA9541EDC7120F66B342D2A451190EB2EE0AC0182D8B139B4EF94` |

The EXE embeds the current model, deployment metadata, Python runtime, frontend
and 3D assets, runtime configuration, BaySpec x86 acquisition helper, and the
vendor user-mode SDK DLL. No neighboring `_internal` directory or Python
installation is required. Run the offline package check before connecting
hardware:

```powershell
& '.\TOUCH-Stable-v0.19.25-Windows-x64.exe' --self-test
```

The matching BaySpec Windows USB driver and the PX6D serial/USB driver remain
system-level prerequisites. See [docs/PORTABLE_RELEASE.md](docs/PORTABLE_RELEASE.md)
for the transfer checklist and
[the v0.19.7-to-v0.19.25 iteration report](docs/TOUCH_STABLE_V0197_TO_V01925_ITERATION_REPORT_20260903.md)
for the evidence-bounded comparison.

## Retained Beta - v0.19.25-beta

The validated Beta package remains installed as a separate release channel.
The v0.19.23 executable and its prior model are also retained as the immediate
rollback path.

## Diagnostics Measurement Analysis

Diagnostics includes a separate `Measure` workspace for completed synchronized
recordings. It compares the optical-only `Fz` estimate against timestamp-aligned
PX6D `Fz`, plots both traces, and reports paired samples, detected loading
cycles, MAE, RMSE, estimated lag, acquisition rate, model inference latency,
and release recovery. Recordings without a force stream remain usable for
optical inspection, but force-consistency metrics are explicitly skipped.

The workspace is manual and offline: `Refresh` scans the selected recording
folder and `Analyze` processes the chosen saved session. It does not poll in the
background, start hardware, modify the Operator view, or use PX6D as a runtime
model input. The same analysis is available from
`scripts/analyze_measurement_session.py` for reproducible report generation.

## Spectrum Normalization

TOUCH can display and record a wavelength-aligned no-contact ratio:

```text
normalized_intensity_ratio(lambda,t) = I(lambda,t) / I0(lambda)
```

`I0` is the median full spectrum accepted by the existing stable
post-release/no-contact baseline gate. The normalization control remains in
Settings. Before an accepted baseline exists, the display falls back to the
processed or raw spectrum and reports that normalization is waiting.

The deployed ordinary-FBG recognition model still receives the original raw
512-point intensity array. Synchronized recordings retain raw counts and add
the aligned `I0` plus `I/I0` when available. Per-frame min-max normalization is
not used because it would erase absolute response evidence and amplify weak
noise. The future mFBG profile retains its separate channel-level `I_i/I0_i`
demodulation contract.

## Recognition Contract

```text
current 512-point full spectrum
  + fixed current-session settled baseline
  + 264 baseline-relative complete-spectrum features
  + 75 joint features from all nine FBG candidates
  -> LightGBM optical contact detector
  -> random-forest 9-position classifier
  -> histogram-gradient-boosting optical Fz estimator
     (unbounded display; approximately 0-6.98 N training range)
  -> continuous digital-twin response
```

The main model domain is manual pressing: approximate position and a broad
fingertip contact area. Push-pull-gauge captures use a much smaller tip at an
exact FBG point and remain a separate reference domain. They are not pooled
with the primary model.

The available no-contact files were generally captured after a press was
released and the sensor recovered. They are therefore recovery-state
baselines, not ideal cold-start baselines. Live inference requires the user to
release contact, wait for a stable spectrum, and set a multi-frame baseline.

## Current Stable Results

Evaluation groups complete acquisition sessions; random adjacent-frame splits
are not used. Stable v0.19.25 contains exactly one hash-bound current model.

| Output | Selected Stable model | Grouped result |
| --- | --- | ---: |
| contact | LightGBM, full spectrum + joint nine-FBG | 96.29% macro-F1 |
| position | random forest, joint nine-FBG | 98.06% accuracy / 98.05% macro-F1 |
| optical Fz | histogram gradient boosting | 0.310 N MAE / 0.929 R2 |

The exact same-day runtime replay produced zero activations in 10,589 dedicated
idle frames, 99.85% active-contact coverage, and zero emitted wrong-position
frames. Blind1, Blind3, and Blind4 are answer-known development data in this
release, so these replay figures are deployment verification rather than
independent blind evidence. A new-date external validation remains required.

## Scientific Boundary

- The Stable optical model outputs an experimental `Fz` estimate learned from
  synchronized PX6D supervision; it is not a certified force measurement.
- Runtime output and the force legend have no fixed upper ceiling. Values above
  approximately 6.98 N are outside the current training range and require higher
  force PX6D data before they can be treated as quantitatively reliable.
- The optical model does not output calibrated pressure or displacement.
- `PX6D Reference Fz` is an independently measured ground-truth label in N;
  it is not an optical-model force prediction.
- Static snapshots do not support tap, slide, or release-dynamics recognition.
- The visual surface is a digital-twin proxy driven by model output, not a
  measured pressure field.
- The current trained model covers 161 same-day sessions and remains pending
  new-date independent validation.

## Layout

| Path | Purpose |
| --- | --- |
| `bayspec_wavelength_shift_app/` | Desktop launcher, local API, BaySpec acquisition, and UI |
| `models/` | Deployed model plus non-deployed candidate bundles |
| `src/hybrid_spectrum/` | Dataset features and runtime model adapter |
| `src/mfbg_intensity/` | Isolated future mFBG 3x3 intensity profile and recorder adapters |
| `scripts/record_live_shadow_comparison.py` | Same-frame deployed/candidate logger |
| `scripts/run_guided_live_shadow_validation.py` | Interactive 9-position/3-level live validation |
| `config/` | Acquisition, baseline, array, and scene configuration |
| `config/mfbg_intensity_3x3.yaml` | Preliminary mFBG channel, intensity, baseline, and coupling contract |
| `tests/` | Demodulation, array orientation, baseline gate, and trained-model tests |

## Run

```powershell
cd bayspec_wavelength_shift_app
run_desktop.bat
```

For the packaged Windows build, download and run the single file
`TOUCH-Stable-v0.19.25-Windows-x64.exe` from the v0.19.25 release. The model,
frontend, configuration, SDK helper, and user-mode SDK DLL are embedded.

Generate the full guided validation plan without touching hardware:

```powershell
.\.venv\Scripts\python.exe scripts\run_guided_live_shadow_validation.py --plan-only
```

When the SDK is live and the sensor is available, omit `--plan-only`. The tool
requires an explicit `RELEASED` confirmation before it clears the old buffer
and captures a new current-session baseline.

Source mode uses `http://127.0.0.1:8640/`. Port 8620 remains reserved for the
optical-intensity application and port 8630 for the provisional
wavelength-shift application.

## Validate

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## PX6D Optical-Force Synchronization

The TOUCH backend can read the PX6D six-axis sensor directly on `COM3` at
921600 baud while BaySpec supplies full-spectrum frames. The Operator summary
shows the conditioned `PX6D Compression Fz`. The Diagnostics **Force**
workspace exposes filtered software-zeroed `Fx/Fy/Fz/Mx/My/Mz`, force and moment resultants,
configured range utilization, tare quality, sample freshness, and explicit
optical-to-mechanical timestamp alignment quality. `Zero Fz` changes only the
software offset and never sends a hardware calibration command.

All six axes pass through a short median despiker and an exponential low-pass
stage. The displayed compression Fz additionally uses a configurable deadband. A slow Fz-only zero-drift tracker is
allowed to move only after the signal has remained stable and close to zero;
it freezes immediately during contact or motion. This conditioning is intended
to suppress unloaded drift without absorbing an applied load. Raw,
software-zeroed, filtered, drift-offset, and drift-corrected values are all
retained in the force export for audit and later calibration. Thresholds are in
`config/px6d_reference.yaml` under `signal`.

To record full optical fingerprints with timestamp-aligned force labels:

```powershell
.\.venv\Scripts\python.exe scripts\record_bayspec_px6d_synchronized.py `
  --base-url http://127.0.0.1:8640 `
  --duration-sec 30 `
  --position P22 `
  --action static_press `
  --trial-id trial_001 `
  --outputs spectrum,response,force `
  --output-root "E:\experiment\TOUCH captures"
```

The user may select any one, two, or all three streams: `spectrum`, `response`,
and `force`. Optical-driven captures use each unique BaySpec host timestamp as
the canonical timeline; the PX6D row is the median six-axis sample inside that
timestamp window. Every selected CSV shares `capture_index`,
`timeline_timestamp_epoch_sec`, and `elapsed_time_sec` exactly. Force-only
capture uses the PX6D host timestamp and does not wait for BaySpec. The three
primary files are `spectrum_timeseries.csv`, `tactile_response_timeseries.csv`,
and `force_timeseries.csv`; unselected files are not created.
`force_timeseries.csv` preserves the original, software-zeroed, and filtered
six-axis PX6D measurements together with the conditioned Fz stages, so
filtering never destroys the calibration source data.

The same workflow is available directly in Diagnostics: select **Force**, set
the position and trial ID, tick the desired streams, and choose a folder with
**Browse**. The UI uses the live continuous `PX6D Fz (N)` reference instead of
a coarse light/normal/hard action label. **Start** and **Stop** create one
session subfolder.
Each synchronized force row stores `force_fz_n` as a continuous regression
target in N; no force-level bins are applied.
`session_metadata.json` includes an automatic cross-file timeline audit.
PX6D remains an independent ground-truth reference; the optical model is not
presented as calibrated force. If no custom folder is chosen, portable desktop
builds default to `data/px6d_synchronized/` beside the executable.

Public research repository for the TOUCH System.
