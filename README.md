# TOUCH System - Trained Static Spectrum Twin

Research software for ordinary-FBG tactile recognition from a complete
512-point BaySpec/Sense spectrum. This edition uses a trained full-spectrum
fingerprint to estimate:

- contact or no contact;
- approximate manual contact position `P11`-`P33`;
- approximate manual response level `light`, `normal`, or `hard`.

It is a separate application. It does not replace or modify the retained PD
voltage, optical-intensity, or provisional wavelength-shift editions.

## Latest Update - v0.15.0

The whole-hand digital twin now preserves the existing modified TOUCH thumb and
adds fitted sensing surfaces to the other four fingertips. The index, middle,
ring, and little fingertip seats are generated from provisional SolidWorks CAD,
aligned to each local fingertip plane, enlarged within a structural border, and
cut to an equal fitted depth. Source STL seats, a geometry manifest, the
integrated whole-hand GLB, and the reproducible integration tool are included.

The current PX6D reconnect, synchronized optical-force recording, dedicated
Record workspace, compact Operator UI, and trained static-spectrum runtime are
unchanged. See [CHANGELOG.md](CHANGELOG.md) for the release summary.

## Recognition Contract

```text
current 512-point full spectrum
  + stable current-session post-release recovery baseline
  -> contact detector
  -> manual-domain 9-position classifier
  -> position-conditioned light/normal/hard classifier
  -> broad fingertip digital-twin response patch
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

| Output | Selected model | OOF result |
| --- | --- | ---: |
| contact | Logistic Regression + current spectrum shape | 100.00% accuracy |
| position | Extra Trees + current spectrum shape | 100.00% accuracy |
| response level | position-conditioned Extra Trees | 97.78% macro-F1 |
| position + level | hierarchical output | 97.78% joint accuracy |

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

- `light`, `normal`, and `hard` are approximate operator response levels.
- The optical model does not output calibrated force, pressure, or displacement.
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
| `scripts/record_live_shadow_comparison.py` | Same-frame deployed/candidate logger |
| `scripts/run_guided_live_shadow_validation.py` | Interactive 9-position/3-level live validation |
| `config/` | Acquisition, baseline, array, and scene configuration |
| `tests/` | Demodulation, array orientation, baseline gate, and trained-model tests |

## Run

```powershell
cd bayspec_wavelength_shift_app
run_desktop.bat
```

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
