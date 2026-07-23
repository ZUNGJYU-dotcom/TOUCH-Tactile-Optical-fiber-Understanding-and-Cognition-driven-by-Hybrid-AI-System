# TOUCH System - Trained Static Spectrum Twin

Standalone BaySpec desktop application for trained ordinary-FBG static
spectrum recognition. It is separate from the PD-voltage, optical-intensity,
and provisional wavelength-shift applications.

## v0.14.0 Desktop Update

Diagnostics now has a dedicated **Record** workspace placed immediately after
**Signal**. **Input** only shows acquisition and adapter status. The Record
workspace keeps the complete synchronized capture workflow in one place, and
the seven-item diagnostic navigation scrolls horizontally when the panel is
narrow. The selected workspace is automatically scrolled into view.

## Robot Hand Scene

The 3D presentation supports three geometry modes: the complete Robot Nano
Hand, the modified thumb holder, and the isolated sensor surface. The complete
hand body is derived from The Robot Studio's MIT-licensed
[Robot Nano Hand](https://github.com/TheRobotStudio/robot-nano-hand) assembly.
Its original thumb tip is omitted and replaced by this project's existing
modified thumb sensor, preserving the calibrated slot and response-surface
alignment. Geometry mode can be changed in Settings or Diagnostics.

## Input and Output

Input is one synchronized 512-point full spectrum plus a stable current-session
baseline. The model outputs contact state, approximate manual contact position
`P11`-`P33`, and approximate `light`/`normal`/`hard` response level.

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

Source server:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8640
```

The desktop launcher embeds the same local UI.

Before connecting hardware on a new computer, the portable desktop build can
verify its bundled frontend, backend contract, SDK helper, and model artifacts
without opening a window, binding port `8640`, or starting PX6D acquisition:

```powershell
& '.\TOUCH System - Trained Static Spectrum Twin.exe' --self-test
```

The process exits with code `0` when every check passes. Details are written to
`%LOCALAPPDATA%\TouchSystemTrainedStaticSpectrumTwin\logs\desktop_launcher.log`.

## Key API

- `GET /api/health`
- `GET /api/global_spectrum_frame`
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
- `tactile_response_timeseries.csv`: contact, position, and light/normal/hard
  temporal-model outputs and probabilities;
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
no custom path is chosen, `data/px6d_synchronized/` is created beside the
portable executable. Browser-only development mode accepts a manually entered
absolute path.

This force is an independent ground-truth measurement used to synchronize and
label BaySpec optical fingerprints. It is not a force estimate produced by the
optical model.

## Limitations

- Current evaluation is within one acquisition session.
- Static snapshots cannot train tap, slide, or release dynamics.
- `light`, `normal`, and `hard` are approximate manual response levels.
- The optical model does not produce calibrated strain, displacement,
  pressure, or force. PX6D Fz is displayed separately as reference truth.
- The visual surface is model-driven and is not a measured pressure map.
- Cross-session collection and validation are still required.
