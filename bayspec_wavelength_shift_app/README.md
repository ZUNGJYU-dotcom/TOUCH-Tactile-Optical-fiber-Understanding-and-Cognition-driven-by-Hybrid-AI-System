# TOUCH System - Trained Static Spectrum Twin

Standalone BaySpec desktop application for trained ordinary-FBG static
spectrum recognition. It is separate from the PD-voltage, optical-intensity,
and provisional wavelength-shift applications.

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
D:\anaconda\miniconda3\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8640
```

The desktop launcher embeds the same local UI.

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
exposes `PX6D Reference Fz` in the Surface Summary. Press `Zero Fz` only while
the sensor is unloaded and stable. The button applies a software tare; raw
six-axis measurements remain unchanged.

Diagnostics has a dedicated **Force** workspace. It shows software-zeroed
`Fx/Fy/Fz/Mx/My/Mz`, resultant force, shear force, resultant moment, configured
range utilization, zero noise, sample age, and firmware. The adjacent sync card
reports the optical frame, force sequence, timestamp offset, aggregation method,
sample count, and an `excellent/good/acceptable/poor` alignment grade. A missing
optical frame is shown as `NO FRAME`; it does not make the independently running
PX6D invalid.

The same workspace provides in-app synchronized recording. Choose a position,
action, and trial ID, then select **Start linked recording**. Each unique BaySpec
spectrum is paired by host timestamp with median PX6D samples and written below
`data/px6d_synchronized/`. **Stop & save** finalizes:

- `synchronized_frames.jsonl`: complete wavelength/intensity arrays plus raw and
  software-zeroed six-axis reference data;
- `frame_summary.csv`: compact flat fields suitable for model training and QA;
- `session_metadata.json`: labels, counts, paired-frame ratio, paths, and sync
  semantics.

In the portable desktop build, this `data/px6d_synchronized/` directory is
created beside the executable. The Diagnostics panel always shows the exact
session directory, so an experiment folder can be located without opening the
application internals.

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
