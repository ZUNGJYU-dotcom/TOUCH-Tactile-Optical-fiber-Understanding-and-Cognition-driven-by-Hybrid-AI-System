# TOUCH

Standalone BaySpec desktop application for ordinary-FBG all-data optical
recognition and experimental `Fz` estimation. It is separate from the
PD-voltage, optical-intensity, and provisional wavelength-shift applications.

## Future mFBG Profile

The ordinary-FBG model remains the active application profile. TOUCH also
loads an isolated future `mfbg_intensity_3x3` profile for nine-channel
spectral-window optical-intensity demodulation. Diagnostics shows this profile
boundary, while the Operator view and existing ordinary-FBG inference remain
unchanged.

Real mFBG 3x3 mode is disabled until measured wavelengths, a real baseline,
and new mFBG calibration data are available. The profile does not output
calibrated force or pressure.

## v0.18.5 Beta

The Beta package contains only the latest optical all-data model. Temporal
optical context drives contact and nine-position recognition; a current-frame
optical model estimates `Fz` in the current 0-5 N research range. PX6D force is
training and validation supervision, never a runtime model feature.

The demonstration replays synchronized real 512-point BaySpec frames. Its nine
peaks are automatically discovered from the no-contact median spectrum and
tracked locally. P11-P33 labels follow provisional ascending wavelength order,
not the 3x3 spatial rendering order and not a final physical channel map.

The automated suite passes with 467 tests and 172 subtests.

## Full-Spectrum Normalization

The spectrum display can use a wavelength-aligned no-contact ratio:

```text
I_normalized(lambda,t) = I(lambda,t) / I0(lambda)
```

`I0` is accepted only from the existing stable multi-frame post-release
no-contact baseline. Until that reference is ready, the UI falls back to the
processed or raw spectrum and reports `I/I0 waiting`. The recognition model
continues to use raw intensity; the display switch cannot redirect model input.

When Spectrum is selected for synchronized recording, each wavelength row
contains raw counts and, when available, aligned baseline counts plus the
normalized ratio. Low or missing reference regions are guarded, and no
per-frame min-max scaling is performed.

## Five-Finger Robot Hand Scene

The 3D presentation supports three geometry modes: the complete Robot Nano
Hand, the modified thumb holder, and the isolated sensor surface. The complete
hand body is derived from The Robot Studio's MIT-licensed
[Robot Nano Hand](https://github.com/TheRobotStudio/robot-nano-hand) assembly.
Its original thumb tip is omitted and replaced by this project's existing
modified thumb sensor, preserving the calibrated slot and response-surface
alignment. The index, middle, ring, and little fingertips each include a carved
sensor recess and a fitted copy of the same tactile response unit. A five-finger
selector scopes the spectrum title, 9-FBG fingerprint, response map, and status
labels to Thumb, Index, Middle, Ring, Little, or All. Simulated response remains
synchronized across all five fingertips while the selector changes the
inspection scope. In the whole-hand view, clicking any fingertip sensor region
or choosing a finger from the selector starts a smooth product-style inspection
of the sensor back side. The translucent previous/next controls move between
adjacent fingertips, while orbit rotation and zoom remain available after the
transition. Geometry mode can be changed in Settings or Diagnostics.

## Input and Output

Input is one synchronized 512-point full spectrum, trailing optical context,
and a stable current-session baseline. The model outputs contact state,
approximate manual contact position `P11`-`P33`, and experimental optical `Fz`.

Manual pressing is the deployment domain: position is approximate and the
fingertip contact area is broad. Push-pull-gauge captures are small-tip exact
point loads and are retained only as a separate reference domain.

## Baseline Procedure

The training no-contact samples are post-press release/recovery spectra, not
ideal cold-start references. For live use:

1. Start SDK acquisition or Watch mode.
2. Release the sensor completely.
3. Wait until the spectrum is visibly stable.
4. Press `Set baseline`; the app requires at least 20 recent frames spanning at
   least 0.6 s; the unified model endpoint requests 30 frames and stores their
   median full spectrum.
5. Apply a static manual press and observe the trained position and response
   level.

The baseline gate reports sample count, time span, noise ratio, drift ratio,
and recovery-baseline status. An unstable or too-short baseline cannot drive
the trained digital twin.

## Local Run

```powershell
run_desktop.bat
```

## Windows Portable Package

Download `TOUCH-v0.18.5-beta-windows-x64.zip`, extract the complete `TOUCH` folder,
and run:

```powershell
.\TOUCH\TOUCH.exe
```

Keep the executable inside the extracted folder because the adjacent runtime,
models, geometry, and frontend assets are required.

Source server:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8640
```

The desktop launcher embeds the same local UI.

Before connecting hardware on a new computer, the portable desktop build can
verify its bundled frontend, backend contract, SDK helper, and model artifacts
without opening a window, binding port `8640`, or starting PX6D acquisition:

```powershell
& '.\TOUCH.exe' --self-test
```

The process exits with code `0` when every check passes. Details are written to
`%LOCALAPPDATA%\TouchSystemTrainedStaticSpectrumTwin\logs\desktop_launcher.log`.

## Key API

- `GET /api/health`
- `GET /api/global_spectrum_frame`
- `GET /api/mfbg-intensity/profiles`
- `GET /api/mfbg-intensity/profile`
- `POST /api/mfbg-intensity/baseline`
- `POST /api/mfbg-intensity/analyze-spectrum`
- `GET /api/mfbg-intensity/recording-preview`
- `POST /api/global_candidate_baseline?minimum_frames=30`
- `POST /api/ingest`
- `POST /api/sdk/start`
- `POST /api/sdk/stop`
- `POST /api/export_watch/start`
- `POST /api/export_watch/stop`
- `GET /api/px6d/latest`
- `GET /api/px6d/trace`
- `POST /api/px6d/tare`
- `GET /api/px6d_capture/status`
- `POST /api/px6d_capture/start`
- `POST /api/px6d_capture/stop`

`GET /api/global_spectrum_frame` always exposes the deployed prediction. Add
`include_shadow=true` only during validation to run the v7 fused-shift
agreement candidate under separate fields. The default Operator request does
not run the candidate, so shadow evaluation does not add latency to the main
display. The candidate is shadow-only: its `drives_operator_ui` and
`drives_digital_twin` flags remain false. Use
`scripts/record_live_shadow_comparison.py` from the project root to record
same-frame labeled comparisons before any promotion decision.

## PX6D Reference Force

When the PX6D is available on `COM3`, the backend starts a protocol reader and
exposes `PX6D Compression Fz` in the Surface Summary. Press `Zero Fz` only while
the sensor is unloaded and stable. The button applies a software tare; raw
six-axis measurements remain unchanged.

Diagnostics has a dedicated **Force** workspace. It shows median-despiked,
low-pass software-zeroed `Fx/Fy/Fz/Mx/My/Mz`, filtered resultant force, shear
force, resultant moment, configured
range utilization, zero noise, sample age, and firmware. The adjacent sync card
reports the optical frame, force sequence, timestamp offset, aggregation method,
sample count, and an `excellent/good/acceptable/poor` alignment grade. A missing
optical frame is shown as `NO FRAME`; it does not make the independently running
PX6D invalid.

All six axes use the same five-sample median despiker and low-pass smoothing.
The compression Fz display additionally uses a near-zero deadband and
stationary-gated slow zero-drift tracking. Drift tracking is limited to Fz and
is frozen as soon as positive contact or motion is detected. Diagnostics shows the
low-pass value, estimated drift offset, and current filter state. All settings
live in `config/px6d_reference.yaml`; the raw six-axis stream is never replaced.

The same workspace provides selectable synchronized recording. Choose a
position, trial ID, save folder, and any one, two, or all three data streams.
The former coarse action selector has been replaced by the live continuous
`PX6D Fz (N)` reference. **Stop** finalizes the selected primary files:

- `spectrum_timeseries.csv`: full wavelength/intensity samples for every frame;
- `tactile_response_timeseries.csv`: contact, position, optical-force estimate,
  and retained compatibility fields for model auditing;
- `force_timeseries.csv`: raw, software-zeroed, and filtered six-axis PX6D reference data.

The force CSV includes `fx_filtered_n` through `mz_filtered_nm` and additionally
contains `median_reference_fz_n`,
`filtered_reference_fz_n`, `drift_offset_n`,
`drift_corrected_reference_fz_n`, `conditioned_reference_fz_n`, `force_fz_n`,
and filter state fields. `force_fz_n` is the continuous non-negative Fz target
in N; it is not converted into light/normal/hard bins. Use the raw/zeroed
columns for traceable calibration work and `force_fz_n` for regression.

All selected files share the exact `capture_index`,
`timeline_timestamp_epoch_sec`, and `elapsed_time_sec`. Unselected primary CSVs
are not created. `synchronized_frames.jsonl`, `frame_summary.csv`, and
`session_metadata.json` are audit sidecars; metadata records the selected
streams and verifies cross-file timeline equality.

The desktop app's **Browse** control opens a native Windows folder chooser. If
no custom path is chosen, a frozen build stores captures under
`Documents/TOUCH/captures`. Browser-only development mode accepts a manually
entered absolute path.

The PX6D stream is an independent ground-truth measurement used to synchronize
and label BaySpec optical fingerprints. The Beta Operator force value is a
separate optical estimate and does not use PX6D as a runtime feature.

## Limitations

- Current grouped evaluation is not yet a cross-device or cross-fabrication
  generalization result.
- The optical `Fz` estimate is a research calibration against synchronized
  PX6D labels, not a certified force measurement.
- The optical model does not produce calibrated strain, displacement, or
  pressure. Diagnostics retains PX6D as an independent reference when present.
- The visual surface is model-driven and is not a measured pressure map.
- Cross-session collection and validation are still required.
