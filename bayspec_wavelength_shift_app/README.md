# TOUCH

Standalone BaySpec desktop application for ordinary-FBG High Sensitivity /
300 us optical recognition and experimental `Fz` estimation. It is separate from the
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

## v0.19.25 Stable

Stable v0.19.25 is an exact promotion of the hardware-tested v0.19.25 Beta
runtime. It retains the same deployed model SHA-256, joint nine-FBG feature
contract, position and force estimators, release-latch repair, and late-frame
protection.

Stable now has isolated High Sensitivity / 300 us user settings and an
independent copy of the validated contact-state configuration. This prevents
the former Stable 5000 us preference or future Beta tuning from changing the
promoted runtime. The live acceptance run recorded 54 press/release frames
without frame-order regression, ended released, and then remained inactive
for all 31 monitored idle frames.

## v0.19.25 Beta

This maintenance release keeps the v0.19.24 same-day nine-FBG model unchanged
and repairs a live release latch found during hardware testing. Concurrent UI
and Record requests can receive adjacent SDK frames in reverse order; late
frames are now ignored instead of being treated as a new temporal segment.

If the learned contact head remains saturated after release, the runtime can
also clear contact from the measured nine-FBG trajectory. This path requires a
previously observed contact peak, at least 60% full-spectrum recovery, two
quiet physical frames over at least 180 ms, no recent spectral activity, and
low spatial confidence. A stationary held press has no peak recovery and is
therefore not released merely because its position confidence is low. The
fixed clean-session model baseline is not moved by this decision.

Exact-adapter replay of Blind1, Blind3, and Blind4 retained zero dedicated-idle
activations and 100% session decisions. One additional 0.085 N release-tail
frame was suppressed; this is below the 0.1 N meaningful-contact boundary used
for runtime validation. The model artifacts remain unchanged; this validated
runtime is the basis of the v0.19.25 Stable promotion.

## v0.19.24 Beta

This Beta uses one same-day High Sensitivity / 300 us runtime built from 161
complete sessions and 49,853 frames. The source boundary is the two regular
2026-09-02 acquisition batches together with the label-cleaned, answer-known
Blind1, Blind3, and Blind4 captures. Blind2 remains excluded. Because those
blind answers are now part of model selection and fitting, none of these
batches is claimed as independent blind evidence for this release.

Contact and optical Fz use 339 features: the 264-value baseline-relative
complete-spectrum view plus a 75-value joint fingerprint of all nine FBG
candidates. Position uses the 75-value joint nine-FBG fingerprint directly.
The selected estimators are LightGBM for contact, random forest for position,
and histogram gradient boosting for force. A single grating is never
sufficient evidence for contact or position.

Complete-session grouped validation reaches 96.78% contact accuracy, 98.06%
raw position accuracy, and 0.310 N optical-force MAE. The stricter
leave-one-complete-acquisition-batch-out runtime stress test reaches 99.09%
position accuracy, 99.08% macro F1, and zero activations across its dedicated
idle frames. The software still reports optical force above the observed
6.98 N training range without an artificial upper clip, but such values are
unvalidated extrapolations.

An exact, answer-known replay of all 161 sessions produced no activation in
10,589 dedicated-idle frames, 99.85% active-contact recall, and no emitted
wrong-position label. Emitted positions were 100% correct; 44 of 33,526
position frames were intentionally withheld during the two-frame confirmation
period. Meaningful-contact detection had 0 ms median and 90th-percentile gate
delay, while first correct position had 0 ms median and 118 ms 90th-percentile
delay at the recorded cadence.

Promotion is hash-bound to the staged model and exact replay. The immediately
preceding v0.19.23 model and complete executable are retained as rollback
artifacts. These results establish same-day compatibility only; a new-date
live idle and nine-position check is still required.

## v0.19.23 Beta

This Beta deploys the Blind3 incremental full-spectrum model. Its isolated
High Sensitivity / 300 us training set contains 69 complete sessions and
26,862 frames; Blind4 remains excluded as an independent test. Contact uses
histogram gradient boosting, position uses random forest, and optical Fz uses
Extra Trees. Every task consumes the complete 264-feature baseline-relative
nine-FBG fingerprint rather than one grating in isolation.

Predictions for the 23 anonymous Blind4 sessions were hash-frozen before the
answer folders were opened, with zero overlap against the training manifest.
The candidate correctly identified all 18 position sessions and all five
no-contact sessions. It produced no activation in the 766 dedicated idle
frames, covered 96.71% of PX6D-active frames, and reached 97.47% position
accuracy among frames where optical contact was active.

An answer-known deployment check then replayed the same captures through the
complete Beta contact, release, and position-lock state machine. This separate
check retained 23/23 session decisions, 0/766 dedicated-idle activations,
97.20% active-frame display coverage, and zero wrong displayed positions on
active reference frames. It is deployment verification, not a second
independent blind result.

Optical force remains experimental. On Blind4 active frames, MAE is 0.552 N,
Pearson r is 0.877, and the calibration slope is 0.769. Loads above 5 N are
still underestimated even though the application applies no explicit upper
clip. The previous deployed runtime remains available as a hash-verified
rollback, and a live post-deployment check is still required.

## v0.19.22 Beta

This release stabilizes the displayed position throughout one contact episode.
The initial two seconds remain responsive so an onset transient can correct
itself. After that window, a position changes only when the full nine-FBG
fingerprint supports the new point with at least 0.55 confidence, 0.15 margin,
and eight consecutive frames. Ambiguous or weak neighbouring predictions keep
the established point until release instead of blanking the surface or making
a P11 press jump to P13.

The deployed High Sensitivity / 300 us model and its training data are
unchanged. On untouched blind recordings, all 14 sessions retained the correct
session identity and P11 produced 117/117 P11 display votes, compared with
56/117 before the state-machine repair. A separate 27-session runtime replay
showed no cross-position assignments and 99.80% active-frame display accuracy.

Record now also stores the exact position and optical force shown by Operator
for each captured frame. The synchronized CSV fields distinguish the displayed
state from the formal and raw classifier labels, making later model-versus-UI
audits reproducible without discarding the authoritative raw spectrum.

## v0.19.20 Beta

This Beta uses the isolated 2026-09-02 `0.19.19-beta` acquisition batch: 55
synchronized High Sensitivity / 300 us sessions and 24,785 recorded frames.
Ten sessions are no-contact references; every P11-P33 position contributes
five independent trials. No earlier exposure mode or acquisition date is used
to fit this model.

Contact, position, and optical Fz are inferred from the complete 264-feature
baseline-relative nine-FBG fingerprint. Five-fold whole-session validation
reaches 98.44% contact accuracy, 99.63% position accuracy, and 0.209 N force
MAE. The runtime force result remains optical-only and is not capped at the
6.37 N calibration boundary.

The release gate now distinguishes slow genuine recontact from a stationary
release residual. A second press can re-arm from sustained high-confidence
nine-FBG contact and position evidence plus clear growth away from the startup
baseline, without waiting for one large frame-to-frame motion spike. Exact
runtime replay across all 45 labelled sessions displays 17,396/17,424 active
frames correctly (99.84%), including 1,973/1,973 P11 frames, with no
cross-position errors and no false P33 assignment from another position.
Seven visual frames occur in one P23 post-release residual segment; final live
idle and labelled-press verification is therefore still required.

## v0.19.19 Beta

Contact onset now requires a joint nine-FBG spectral fingerprint. Every frame
is evaluated from the complete 512-point spectrum plus the wavelength shift,
integrated area, peak height, and local shape change of all nine gratings. An
isolated peak disturbance can no longer activate the tactile twin. A very
light response is accepted only when the distributed optical evidence and the
full-spectrum contact/position models agree with high confidence.

This change does not replace or retrain the deployed High Sensitivity / 300 us
model. It adds an out-of-distribution guard before contact is latched, while
the existing baseline-separated plateau logic continues to preserve a real
stationary press and confirmed recovery still clears release immediately.

Exact runtime replay over all 45 labelled position sessions retained 0 visual
false activations in 1,815 idle frames and reached 99.835% position display
accuracy over 11,513 active frames. P23 was correct for all 1,362 active P23
frames, no other position was displayed as P23, and all 11,243 sustained
interior frames across P11-P33 were correct. The 19 unshown frames occurred
only at 0.25-0.5 N press/release boundaries; every active frame above 0.5 N was
correctly displayed. A separate 68-frame current-hardware idle replay also
produced no activation.

## v0.19.18 Beta

The runtime now accepts only an explicitly tagged no-contact baseline collected
from the current application session. A generic or post-release reference can
no longer silently become the startup model baseline, which prevents optical
source drift from turning P11/P21 into P13/P33 predictions.

Sustained, baseline-separated contact remains latched through stationary
plateaus and brief classifier dips. A confirmed release transition clears the
operator surface immediately instead of waiting for a low-noise re-anchoring
window; partial relaxation below the configured 45% recovery threshold is not
treated as release.

An exact Beta-gate replay over all 45 High Sensitivity / 300 us position
sessions covered 11,513 active frames and 1,815 idle frames. Idle visual false
activation was 0%, active position display accuracy was 99.887%, and no
P11-P33 cross-position confusion was observed. Every position achieved 100%
accuracy on the sustained interior of its press intervals; the remaining 13
unshown active frames occurred only at confirmation or release boundaries.

## v0.19.17 Beta

The Beta runtime now uses only the isolated 2026-09-02 High Sensitivity / 300
us model. Its source set contains 55 synchronized sessions and 17,254 frames:
ten no-contact sessions plus five independent trials at every P11-P33
position. Earlier dates remain reference-only and were not used for fitting or
selection.

Five-fold whole-session validation produced 98.05% contact accuracy, 99.44%
position accuracy, 0.181 N optical-force MAE, and 0.971 R2. The calibrated data
extend to 6.04 N; runtime force output has no fixed upper clip. A separate
current-session-baseline gate replay produced no visual idle activation and
99.86% active position accuracy. That replay checks runtime compatibility on
the training domain; the grouped out-of-fold figures are the generalization
evidence.

The startup baseline must be collected while the sensor is unloaded. A stale
reference was deliberately tested and degraded recognition after optical
drift, so refresh the baseline or restart TOUCH after the source operating
point changes.

## v0.19.16 Beta

Record now writes every available unique BaySpec spectrum and synchronized
PX6D Fz sample without waiting for the recognition model. An exact same-frame
cached prediction is retained when available; otherwise its response row is
marked `capture_response_deferred` and can be reconstructed from the saved raw
spectrum offline. No prediction from another frame is substituted.

The recorder polls at 10 ms and commits data in small durable batches. A local
512-point spectrum + recognition + force benchmark sustained 43.2 frames/s;
the Diagnostics Record panel shows the measured frame rate for each session.
The model baseline is fixed after the current-session startup baseline settles,
so rest recovery cannot shift the position model's reference coordinates.
The Beta runtime contains exactly one deployed model. Its runtime identity is
`ordinary_fbg_high_sensitivity_300us_beta_v3`; older all-data and three-date
models are not bundled or exposed as selectable recognition paths.

## v0.19.7 Stable

The v0.19.7 Stable and Beta packages replace corrupted source-status separators
with ASCII text such as `SDK | idle`. Model weights, acquisition logic, and
hardware interfaces are unchanged. The Stable package
contains one deployed model and excludes every legacy static or dynamic
inference fallback.

Each new live or watched acquisition establishes a five-frame stable baseline
from the current session before model output can drive the tactile surface.
Low-force contact and position changes require two consecutive physical
spectra. The nine-position replay regression produced no idle visual
activation and no non-P23-to-P23 display errors with this baseline policy.

## Historical v0.19.7 Notes

The former v0.19.7 Beta package was retained locally during its release cycle as
a rollback build. Its Beta and Stable variants used the same model and inference
logic while retaining separate release manifests and shortcuts. These notes are
historical and do not describe the current v0.19.25 Stable runtime.

Data recording includes a compact Force Sensor `Zero` command. It shares the
PX6D software-zero state with the Force and Diagnostics views and refreshes the
recording readiness cells when zeroing completes. Start is now actionable
before setup is complete so it can identify the missing requirement, while the
backend still blocks incomplete capture setup.

That release contained one optical all-data model. Temporal optical context
drove contact and nine-position recognition, while a current-frame optical
model estimated `Fz`. PX6D force was training and validation supervision rather
than a runtime model feature.

Brief spatial uncertainty now holds the current contact instead of releasing
it. A quiet residual is cleared only after about one second without fresh
spectral activity or a credible spatial fingerprint, and slow baseline drift
alone cannot reactivate the surface. Provisional visual positions are locked
through quiet classifier jitter until a new spectral event is observed.

The demonstration replays synchronized real 512-point BaySpec frames. Its nine
peaks are automatically discovered from the no-contact median spectrum and
tracked locally. P11-P33 labels follow provisional ascending wavelength order,
not the 3x3 spatial rendering order and not a final physical channel map.

The automated suite passes with 490 tests and 172 subtests.

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

Stable v0.19.25 is published as a true single-file application:

```powershell
.\TOUCH-Stable-v0.19.25-Windows-x64.exe
```

The published file is 103,285,538 bytes (98.50 MiB), with SHA-256
`7963470D59CDA9541EDC7120F66B342D2A451190EB2EE0AC0182D8B139B4EF94`.

It embeds the current v0.19.25 model and deployment record, frontend and 3D
assets, Python runtime, Stable-specific configuration, BaySpec x86 acquisition
helper, and vendor user-mode SDK DLL. No adjacent `_internal` folder or Python
installation is required.

Source server:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8640
```

The desktop launcher embeds the same local UI.

Before connecting hardware on a new computer, the portable desktop build can
verify its bundled frontend, backend contract, SDK helper, and model artifacts
without opening a window, binding port `8640`, or starting PX6D acquisition:

```powershell
& '.\TOUCH-Stable-v0.19.25-Windows-x64.exe' --self-test
```

The process exits with code `0` when every check passes. Details are written to
`%LOCALAPPDATA%\TOUCH\logs\desktop_launcher.log`.

The target computer still needs the matching BaySpec Windows USB driver, and
PX6D use requires its serial/USB driver. These system-level drivers are not
silently installed by the application package. See
`docs/PORTABLE_RELEASE.md` for the full transfer checklist.

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

Raw spectrum and PX6D rows are the authoritative training evidence. Recognition
never delays acquisition: `capture_response_source` identifies an exact
same-frame cache hit, while `capture_response_deferred_reason` records why a
prediction should be recomputed offline. Writes are durably synchronized every
10 frames or 250 ms and always flushed when **Stop** completes.
When Recognition is selected without the Spectrum CSV, the lossless JSONL
sidecar still retains the matching raw spectrum required for reconstruction.

The desktop app's **Browse** control opens a native Windows folder chooser. If
no custom path is chosen, a frozen build stores captures under
`Documents/TOUCH/captures`. Browser-only development mode accepts a manually
entered absolute path.

The PX6D stream is an independent ground-truth measurement used to synchronize
and label BaySpec optical fingerprints. The Stable Operator force value is a
separate optical estimate and does not use PX6D as a runtime feature.

## Limitations

- Current grouped evaluation is not yet a cross-device or cross-fabrication
  generalization result.
- The optical `Fz` estimate is a research calibration against synchronized
  PX6D labels, not a certified force measurement.
- Values above 5 N are shown without software clipping but remain unvalidated
  until higher-force synchronized PX6D data are collected and evaluated.
- The optical model does not produce calibrated strain, displacement, or
  pressure. Diagnostics retains PX6D as an independent reference when present.
- The visual surface is model-driven and is not a measured pressure map.
- Cross-session collection and validation are still required.
