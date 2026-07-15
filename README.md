# TOUCH System - Trained Static Spectrum Twin

Private research software for ordinary-FBG tactile recognition from a complete
512-point BaySpec/Sense spectrum. This edition uses a trained full-spectrum
fingerprint to estimate:

- contact or no contact;
- approximate manual contact position `P11`-`P33`;
- approximate manual response level `light`, `normal`, or `hard`.

It is a separate application. It does not replace or modify the retained PD
voltage, optical-intensity, or provisional wavelength-shift editions.

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
- No calibrated pressure, displacement, or `force_N` is output.
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

Private repository. All rights reserved.
