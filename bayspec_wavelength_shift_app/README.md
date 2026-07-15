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

`GET /api/global_spectrum_frame` always exposes the deployed prediction. Add
`include_shadow=true` only during validation to run the v7 fused-shift
agreement candidate under separate fields. The default Operator request does
not run the candidate, so shadow evaluation does not add latency to the main
display. The candidate is shadow-only: its `drives_operator_ui` and
`drives_digital_twin` flags remain false. Use
`scripts/record_live_shadow_comparison.py` from the project root to record
same-frame labeled comparisons before any promotion decision.

## Limitations

- Current evaluation is within one acquisition session.
- Static snapshots cannot train tap, slide, or release dynamics.
- `light`, `normal`, and `hard` are approximate manual response levels.
- No calibrated strain, displacement, pressure, or `force_N` is produced.
- The visual surface is model-driven and is not a measured pressure map.
- Cross-session collection and validation are still required.
