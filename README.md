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
| `models/` | V4 trained static full-spectrum model bundle |
| `src/hybrid_spectrum/` | Dataset features and runtime model adapter |
| `config/` | Acquisition, baseline, array, and scene configuration |
| `tests/` | Demodulation, array orientation, baseline gate, and trained-model tests |

## Run

```powershell
cd bayspec_wavelength_shift_app
run_desktop.bat
```

Source mode uses `http://127.0.0.1:8640/`. Port 8620 remains reserved for the
optical-intensity application and port 8630 for the provisional
wavelength-shift application.

## Validate

```powershell
D:\anaconda\miniconda3\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Private repository. All rights reserved.
