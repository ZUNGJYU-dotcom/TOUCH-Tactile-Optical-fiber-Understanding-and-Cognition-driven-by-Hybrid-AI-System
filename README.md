# TOUCH

Research software for optical-fiber tactile sensing. The current Stable Operator
runtime uses a trained ordinary-FBG 512-point BaySpec/Sense all-data model to
estimate:

- contact or no contact;
- approximate manual contact position `P11`-`P33`;
- continuous optical `Fz` with no fixed software display/output ceiling; the
  current model has only been validated on 0-5 N supervision.

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

## Latest Update - v0.19.7 Stable

The v0.19.7 Stable and Beta packages remove a corrupted source-status separator
that could render as a Chinese character in labels such as `SDK idle`. Source
states now use ASCII text such as `SDK | idle`; models, acquisition, and
recognition behavior are unchanged. Stable uses the
same three-date optical model, continuous optical `Fz` estimator, interface,
and device contracts as Beta. Its portable package contains only the current
deployed model and has no legacy inference fallback.

Every new live or watched acquisition now clears cross-session optical
references and establishes its baseline from five stable spectra in the
current session. Two consecutive physical spectra are required before a
low-force visual response or a new position can drive the digital twin. In the
nine-position replay audit this removed idle visual activation and eliminated
non-P23 inputs being displayed as P23 without changing model weights.

Data recording now exposes a local `Zero` command beside the Force Sensor
value. It uses the same PX6D software zero as the Force and Diagnostics views,
then immediately refreshes recording readiness. Start remains clickable while
setup is incomplete and reports the exact missing requirement; it still
refuses to record until the selected position, requested streams, live optical
frame, force zero, and destination folder are valid.

The Stable runtime deploys the latest all-data optical bundle: temporal optical
context supports contact and nine-position inference, while the current-frame
optical head estimates `Fz`. Grouped evaluation is by `session_id`; force-sensor
measurements are never model inputs.

The live runtime now requires fresh full-spectrum activity or credible spatial
evidence before accepting a learned contact prediction. Stable residual drift
is therefore returned to `no_contact`, with optical force reset to zero and no
position emitted. Position probabilities are smoothed and checked for
confidence and margin before the digital twin moves to another array location.
After a stationary release, the runtime reference can be re-anchored without
changing the trained model.

Low position confidence is no longer an immediate release signal. Brief
uncertainty preserves a real contact, while a stationary residual with no new
spectral activity and no credible spatial fingerprint returns to `no_contact`
after about one second. Slow baseline drift alone cannot re-arm contact. A
provisional visual location is established only around real spectral activity
and held through quiet classifier jitter; the formal position result remains
unaccepted and no unknown contact is forced to P22.

The built-in demonstration now replays synchronized real 512-point BaySpec
spectra. Its nine spectral peaks are discovered automatically from the robust
no-contact median and tracked locally in every frame. The provisional spectral
assignment follows ascending wavelength order:

```text
P11, P12, P13, P21, P22, P23, P31, P32, P33
```

This wavelength-order assignment is separate from the 3x3 screen layout and is
not a final measured physical channel map. The complete automated suite passes
with 490 tests and 172 subtests.

See [CHANGELOG.md](CHANGELOG.md) for the complete release summary.

## Retained Beta - v0.19.7-beta

The validated Beta package is retained locally as a rollback build. Stable and
Beta use the same model and runtime logic; their release manifests and desktop
shortcuts remain separate.

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
  + trailing optical context
  + stable current-session post-release recovery baseline
  -> optical contact detector
  -> optical 9-position classifier
  -> optical Fz estimator (unbounded display; 0-5 N validated range)
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

## Current Baseline Results

Evaluation uses leave-one-repeat-index-out groups over independent static CSV
files. Random snapshot splitting is not used.

| Output | Selected Stable model | Grouped result |
| --- | --- | ---: |
| contact | temporal Extra Trees | 94.13% macro-F1 |
| position | temporal Extra Trees | 98.85% macro-F1 |
| optical Fz | current-frame Extra Trees | 0.268 N MAE |

These are single-session baseline results. They are not a final cross-session
generalization claim. The position model was stable across the available early
and late recovery-baseline fixtures; force-level inference remains more
baseline-sensitive and needs a fresh stable baseline in each session.

## V7 Fused-Shift Agreement Candidate

The deployed bundle remains unchanged. A separate v7 candidate runs beside it
in shadow mode and cannot control the Operator UI or digital-twin deformation.
Its position head uses only the nine signed, per-frame-normalized fused
common-mode-corrected FBG shifts. Logistic Regression, shrinkage LDA, and a
linear SVM vote on the position. Reported position confidence is the
uncalibrated member vote fraction, not a probability.

Strict old-to-new capture evaluation produced 0.908 position macro-F1; the
reverse new-to-old challenge produced 0.902. Unanimous position votes covered
81.1% of challenge spectra and were 95.4% accurate. Response-level macro-F1
was 0.792, with hard-response recall 1.000; cross-session light/normal scaling
is still unresolved. The candidate therefore requires labeled live validation
and has not been promoted.

For auditable live comparison:

```powershell
.\.venv\Scripts\python.exe scripts\record_live_shadow_comparison.py `
  --base-url http://127.0.0.1:8640 `
  --duration-sec 10 `
  --expected-position P22 `
  --expected-force normal
```

Baseline capture is never implicit. The optional `--set-baseline-first` flag
also requires `--confirm-sensor-released`.

The Operator UI uses the default `GET /api/global_spectrum_frame` request,
which does not run the candidate. Shadow inference is opt-in through
`include_shadow=true`; the validation script sets this flag automatically.
This keeps candidate evaluation from adding latency to the deployed display.
The shadow result also includes a five-unique-frame diagnostic vote, with
three contact frames required and two release frames required. It remains
diagnostic-only and cannot drive the UI or deformation.

## Scientific Boundary

- The Stable optical model outputs an experimental `Fz` estimate learned from
  synchronized PX6D supervision; it is not a certified force measurement.
- Runtime output and the force legend have no fixed 5 N ceiling. Values above
  5 N are explicitly outside the current validated range and require higher
  force PX6D data before they can be treated as quantitatively reliable.
- The optical model does not output calibrated pressure or displacement.
- `PX6D Reference Fz` is an independently measured ground-truth label in N;
  it is not an optical-model force prediction.
- Static snapshots do not support tap, slide, or release-dynamics recognition.
- The visual surface is a digital-twin proxy driven by model output, not a
  measured pressure field.
- The current trained model is a single-session baseline pending multi-session
  validation.

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

For the packaged Windows build, download
`TOUCH-v0.19.7-stable-windows-x64.zip` from the release, extract the complete
`TOUCH` folder, and run `TOUCH\TOUCH.exe`. Do not move only the executable out
of the extracted folder because its bundled runtime and assets are required.
The promoted model is bundled in that release package; the 118 MB joblib is
not stored as an ordinary Git blob because it exceeds GitHub's per-file limit.

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
D:\anaconda\miniconda3\python.exe -m unittest discover -s tests -p "test_*.py" -v
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
