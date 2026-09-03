# Changelog

## Unreleased

## v0.19.25 - 2026-09-03

### v0.19.25 Stable promotion

- Promote the user-validated v0.19.25 Beta runtime without changing the
  deployed model artifact, feature schema, thresholds, or release repair.
- Publish a true Windows x64 single-file Stable executable containing the
  current model, deployment record, frontend, 3D assets, Python runtime,
  configuration, BaySpec x86 helper, and vendor user-mode SDK DLL.
- Give Stable its own High Sensitivity / 300 us processing settings and its
  own copy of the validated contact-state configuration, preventing legacy
  5000 us settings or later Beta experiments from changing Stable behavior.
- Package only the deployed current model and its hash-bound deployment
  metadata. Beta and the v0.19.23 rollback build remain separate.
- Record live validation with 54 press/release capture frames, no frame-order
  regressions, a released final state, and 31 subsequent idle frames with zero
  false activation.

### v0.19.25 Beta release-latch repair

- Ignore late SDK frames that arrive out of order when concurrent Operator and
  Record requests snapshot adjacent spectra. Such frames no longer rewind the
  temporal adapter or erase the physical contact peak used for release.
- Add a conservative nine-FBG residual-recovery release path for cases where
  the contact classifier remains saturated after the finger is removed. It
  requires an armed contact, at least 60% recovery from the observed contact
  peak, a quiet complete spectrum, no recent spectral activity, low position
  confidence, and two physical frames spanning at least 180 ms.
- Preserve the fixed clean-session model baseline and the isolated current-only
  model bundle. The Stable edition, deployed estimator artifacts, and training
  data are unchanged.
- Replay Blind1, Blind3, and Blind4 through the exact adapter with zero
  dedicated-idle activations. The only additional suppressed active frame is a
  0.085 N release-tail sample; all three batches retain 100% session accuracy.

### v0.19.24 Beta same-day joint nine-FBG runtime

- Train one same-day runtime from 161 complete High Sensitivity / 300 us
  sessions and 49,853 frames spanning the two regular batches plus the
  answer-known Blind1, Blind3, and Blind4 batches. Blind2 remains excluded;
  these formerly blind batches are training data now and are not presented as
  independent blind evidence.
- Require the complete optical fingerprint for every task. Contact and force
  use 339 features made from the 264-value baseline-relative full-spectrum
  view plus 75 joint nine-FBG features; position uses the 75-value joint
  nine-FBG view. No task treats one grating as sufficient evidence.
- Deploy LightGBM for contact, random forest for position, and histogram
  gradient boosting for optical Fz. Complete-session grouped evaluation gives
  98.06% raw position accuracy and 0.310 N force MAE; leave-one-acquisition-
  batch-out runtime stress gives 99.09% position accuracy and zero dedicated-
  idle activations.
- Replay all 161 sessions through the exact Beta adapter and state machine:
  0/10,589 dedicated-idle activations, 99.85% active-contact recall, zero
  emitted wrong-position frames, and 100% accuracy whenever a position is
  emitted. Forty-four position frames are deliberately withheld by the
  confirmation gate rather than assigned to another point.
- Bind promotion to the staged-model SHA-256 and exact-replay artifact, archive
  the preceding deployed model before replacement, and keep the complete
  v0.19.23 executable as an immediate desktop rollback. New-date live
  validation remains required.

### v0.19.23 Beta Blind4-validated incremental model

- Extend the isolated High Sensitivity / 300 us training set with the 14
  unblinded Blind3 sessions, for 69 complete sessions and 26,862 frames. The
  newly collected Blind4 sessions remain excluded from training.
- Deploy histogram gradient boosting for contact, random forest for position,
  and Extra Trees for optical Fz, all using the complete 264-feature
  baseline-relative nine-FBG fingerprint.
- Freeze Blind4 predictions before opening its answer folders and verify zero
  overlap with training. The candidate identifies all 18 position sessions and
  all five no-contact sessions, with no activation in 766 dedicated idle
  frames and 97.47% position accuracy when optical contact is active.
- Re-run the deployed model through the complete Beta contact state machine as
  an explicitly marked post-unblind deployment check. It retains 23/23 session
  decisions, 0/766 dedicated-idle activations, 97.20% active-frame display
  coverage, and zero wrong displayed positions on active reference frames.
- Retain optical force as experimental: Blind4 active-frame MAE is 0.552 N and
  forces above 5 N remain underestimated. Preserve the preceding runtime as a
  hash-verified rollback and require a post-deployment live check.

### v0.19.22 Beta position episode lock and synchronized display exports

- Keep the selected position stable for the active contact episode after a
  two-second onset-settling window. Weak or ambiguous neighbouring-class
  fluctuations can no longer move a sustained P11 press to P13; a real
  in-contact move still switches after eight consecutive frames with strong
  nine-FBG evidence.
- Preserve the established position through temporary classifier ambiguity
  until the existing contact-release state machine confirms release. This
  removes visible blanking and relabelling without changing the deployed model
  artifact or its training data.
- Replay the untouched blind set with answer labels excluded from inference.
  All 14 sessions retain the correct session identity, and P11 improves from
  56/117 P11 display votes to 117/117. A separate 27-session training-domain
  state-machine replay contains no cross-position assignments and reaches
  99.80% active-frame display accuracy.
- Extend synchronized Record outputs with the exact contact state, position,
  confidence, margin, and optical force shown by Operator on the same frame,
  while retaining separate formal and raw model labels for comparison.

### v0.19.20 Beta new-data model and gradual recontact repair

- Train an isolated model from the complete 2026-09-02 `0.19.19-beta`
  acquisition batch: 55 synchronized sessions, 24,785 frames, ten no-contact
  sessions, and five independent trials for every P11-P33 position. Earlier
  acquisition modes and dates remain excluded.
- Use all 264 baseline-relative full-spectrum features for contact, position,
  and optical Fz. The selected models are LightGBM, random forest, and Extra
  Trees respectively; whole-session five-fold validation reaches 98.44%
  contact accuracy, 99.63% position accuracy, and 0.209 N force MAE.
- Fix a release-state lock that could suppress a genuine gradual second press
  at high frame rates. A sustained high-confidence nine-FBG position/contact
  signature plus clear movement away from the session baseline can now re-arm
  contact without requiring one large inter-frame motion spike.
- Replay all 45 labelled sessions through the exact Beta state machine. P11
  continuous coverage improves from 1,664/1,973 to 1,973/1,973; aggregate
  active display accuracy is 99.84%, there are no cross-position assignments,
  and no non-P33 sample is displayed as P33. Seven displayed frames remain in
  one P23 post-release residual sequence (0.41% of idle-labelled frames).
- Preserve the previous deployed model as a hash-verified rollback and mark
  this build for final live idle and labelled-press validation.

### v0.19.19 Beta nine-FBG joint contact signature

- Evaluate every contact onset from the complete nine-grating response rather
  than allowing a single peak disturbance to drive the tactile twin. The
  explicit 36-value local signature contains wavelength, area, height, and
  shape changes for all nine FBG candidates alongside the existing 512-point
  full-spectrum classifiers.
- Require a distributed measured response across at least five low-response
  and three nominal-response gratings. Retain a conservative light-contact
  path only when the full-spectrum contact and position models are both highly
  confident.
- Apply the joint signature to formal contact, model-consensus contact, low
  force visual activation, and trusted-rest unlock. Do not re-gate an already
  established stationary plateau, preserving sustained P11-P33 response.
- Replay all 45 High Sensitivity / 300 us labelled sessions: 0/1,815 idle
  visual activations, 11,494/11,513 correct active position displays, no
  cross-position confusion, and 11,243/11,243 correct sustained-interior
  frames. All 19 omitted frames were 0.25-0.5 N boundary samples.
- Keep the deployed v3 model artifact unchanged; this release modifies runtime
  evidence validation only.

### v0.19.17 Beta 2026-09-02 acquisition-domain model

- Train a new isolated High Sensitivity / 300 us candidate from 55 complete
  synchronized sessions recorded on 2026-09-02: ten no-contact sessions and
  five independent trials for each of P11-P33, totalling 17,254 frames.
- Keep every earlier acquisition date out of fitting and candidate selection.
  All formal metrics use five-fold whole-session holdout, never a random frame
  split.
- Select histogram gradient boosting for contact, random forest for position,
  and Extra Trees for continuous optical Fz. Grouped out-of-fold performance is
  98.05% contact accuracy, 99.44% position accuracy, 0.181 N force MAE, and
  0.971 R2 over the measured 0-6.04 N range.
- Preserve unlimited positive force extrapolation instead of clipping at the
  calibration maximum.
- Save the immediately preceding deployed model as a hash-verified rollback on
  every promotion. The deployment tool now accepts explicit candidate and
  dataset directories rather than being tied to the 2026-08-31 run.
- Validate the final runtime gate with a fresh current-session baseline: 0%
  visual idle activation over 1,815 idle frames, 99.86% active position
  accuracy over 11,513 active frames, and no non-P23 false-P23 assignments.
  This gate replay is a training-domain compatibility check; grouped OOF
  metrics remain the independent generalization evidence.
- Retain the fixed model baseline policy. A deliberately stale recorded
  reference degraded the long-duration replay, so live validation must begin
  from the startup-settled no-contact baseline and optical drift requires an
  explicit refresh or application restart.

### v0.19.16 Beta fixed-session baseline and high-rate Record

- Keep the model's startup-settled baseline fixed for the complete live session;
  rest recovery may suppress idle output but cannot rewrite the
  baseline-relative position coordinate system.
- Decouple synchronized raw spectrum/PX6D capture from model inference. Record
  accepts an inference result only when the exact same spectrum frame is
  already cached; otherwise the aligned response row is explicitly marked for
  offline reconstruction instead of blocking capture or reusing stale output.
- Poll Record at 10 ms and durably flush batches every 10 frames or 250 ms.
  A 512-point spectrum + response + force benchmark sustained 43.2 frames/s,
  compared with about 3 frames/s in the previous synchronous inference path.
- Report the measured capture rate in Diagnostics and preserve exact timeline
  equality across all selected CSV streams and the lossless JSONL sidecar.
- Package only the High Sensitivity / 300 us deployed model in Beta and remove
  the remaining all-data/three-date runtime labels from Diagnostics, health
  metadata, and launcher validation. Historical artifacts remain outside the
  Beta runtime for research traceability and Stable rollback only.
- Prevent the fixed-baseline rest lock from entering an automatic-reanchor
  candidate that cannot complete when model-baseline updates are disabled.
  The final 18-session replay produced 0% idle activation across 545 idle
  frames and 99.64% correct nine-position output across 2800 active frames.

### v0.19.15 Beta baseline-response latency repair

- Keep the trusted-rest baseline and formal two-frame contact confirmation,
  while allowing a fresh, baseline-separated, spatially credible optical press
  to arm the operator visual contact path on its first physical frame.
- Retain two-frame position confirmation so a single P23 classifier spike
  cannot capture the tactile surface position.
- Preserve the v0.19.14 grouped replay results: 0.55% idle visual tail rate,
  98.39% active visual coverage, 98.32% position accuracy, and zero non-P23
  false-P23 assignments.

### v0.19.14 Beta intermittent idle-contact repair

- Add a true one-file Windows Beta build containing the only deployed model,
  frontend, configuration, BaySpec x86 helper, and vendor SDK DLL, plus a
  current-user Setup EXE, portable ZIP, and SHA-256 release manifest.
- Latch an operator-confirmed or startup-settled spectrum as trusted rest so
  isolated High Sensitivity noise cannot bypass no-contact through a single
  activity frame.
- Unlock real contact after two physical frames using either credible spatial
  evidence or a baseline-relative spectral departure above the measured idle
  envelope; this retains low-force response while suppressing idle drift.
- Re-anchor moderate stationary rest drift without accepting it as contact,
  and clear stale surface, trace, position, and force visuals immediately when
  Live or Watch acquisition stops.
- Validate against a 600-frame 300 us High Sensitivity no-contact capture with
  zero formal contact, zero visual activation, and zero position output. A
  grouped nine-position press replay retained 98.39% visual coverage and
  98.32% position accuracy using a current-session baseline.

### v0.19.13 Beta isolated 300 us model and low-latency SDK stream

- Deploy the isolated 2026-08-31 High Sensitivity / 300 us model for Beta
  hardware evaluation while preserving the previous runtime as a hash-verified
  rollback artifact; Stable is unchanged.
- Use one persistent x86 BaySpec SDK process for continuous acquisition and
  restart it only after a stream failure or an explicit setting change.
- Reduce the default inter-frame interval from 20 ms to 10 ms. The 300 us
  detector exposure is unchanged and remains adjustable.
- Keep the grouped out-of-session evidence visible as candidate evidence:
  contact accuracy 0.978, position accuracy 0.998, and optical-force MAE
  0.152 N. Live hardware validation remains required.

### v0.19.12 Beta BaySpec high-sensitivity acquisition

- Make BaySpec sensor mode `0` (`High Sensitivity`) the explicit Beta default
  instead of relying on the helper's previous implicit mode selection.
- Set the default detector exposure to `300 us` and expose an exact
  `1-10000000 us` numeric control in Settings.
- Restart an active direct-SDK session when either sensor mode or exposure
  changes, and reject helper frames whose echoed mode/exposure differs from
  the requested acquisition contract.
- Isolate Beta acquisition preferences in
  `%LOCALAPPDATA%/TOUCH/spectrum_processing_beta.json`, leaving Stable's
  existing settings untouched.

### v0.19.7 Stable and Beta status-text hotfix

- Removed the corrupted status separator that rendered as Chinese `路` in source labels such as `SDK idle`.
- Standardized source states as `SDK | idle`, `SDK | live`, and equivalent Watch/HTTP labels.
- Cleaned related mojibake in diagnostics units and separators without changing models, acquisition, or recognition behavior.

### v0.19.6 Stable promotion

- Promote the verified v0.19.6 Beta runtime to Stable without changing model
  weights, optical-force behavior, device interfaces, or Operator UI logic.
- Package only `ordinary_fbg_current_runtime.joblib` and the runtime-required
  configuration files; legacy candidates and fallback models remain excluded.
- Retain the current-session five-frame baseline, two-frame visual contact and
  position confirmation, and post-release baseline recovery that removed the
  observed idle activation and non-P23-to-P23 replay errors.
- Keep the verified Beta package available locally as a rollback build.

### v0.19.6 Beta current-session baseline repair

- Clear cross-session optical references whenever a new live or watched
  acquisition starts, then establish the runtime baseline from five stable
  spectra in the current session before allowing model output to drive the
  digital twin.
- Require two consecutive physical spectra for low-force visual activation
  and a new visual position, preventing an isolated spatial spike from taking
  over the surface display.
- Add nine-position replay regression evidence for idle false activation and
  non-P23-to-P23 confusion without changing the deployed model weights.

### v0.19.4 Beta unified Operator visualization frame

- Drive Surface Summary, the 3D and 2D tactile surfaces, Response Trace, and
  optical force from one versioned Beta inference frame.
- Keep every Operator visualization neutral while the current Beta prediction
  is warming or blocked instead of falling back to legacy model fields or raw
  wavelength proxies.
- Publish synchronized spectrum, surface, trace, summary, and force frame IDs
  for diagnostics without changing the deployed three-date model.

### Unbounded optical-force presentation

- Remove the fixed 5 N runtime and UI ceiling from optical `Fz` output,
  Response Trace, and the Operator force legend.
- Auto-range the force axis upward while retaining 0-5 N as the current
  validated calibration domain; estimates above it are marked unvalidated.
- Keep digital-twin deformation bounded independently for visual stability.

### Diagnostics measurement analysis

- Add a Diagnostics-only `Measure` workspace for saved synchronized optical and
  PX6D recordings; the Operator interface is unchanged.
- Export the current optical-only `Fz` estimate and model inference latency with
  every synchronized recording frame while retaining PX6D `Fz` as reference
  supervision rather than a runtime model input.
- Add offline force-consistency, lag, cycle, repeatability, recovery, acquisition
  cadence, and inference-latency analysis with explicit optical-only handling
  when force reference data were not recorded.
- Keep analysis user-triggered and reject analysis of the session that is still
  being recorded, so the feature does not poll or contend with live hardware.

### v0.18.9 Stable promotion

- Promote the validated v0.18.9 Beta all-data optical model and low-latency
  runtime to the Stable release channel.
- Ship one deployable optical model only, with no legacy static or dynamic
  inference fallback in the Stable package.
- Add a Stable release manifest and a dedicated PyInstaller contract while
  retaining the Beta package as a local rollback build.
- Verify the frozen Windows executable, bundled model hash, SDK helper hash,
  and selected runtime contract before publishing.

### v0.18.9 Beta recording readiness repair

- Add a compact `Zero` command beside the live Force Sensor value in Data
  recording, sharing the same PX6D software-zero state as Force and
  Diagnostics.
- Refresh recording readiness immediately after zeroing so Force changes from
  `Zero` to `Ready` without leaving the recording panel.
- Keep Start actionable while setup is incomplete; clicking it now reports the
  exact missing items instead of remaining silently disabled.
- Preserve the required position label, optical-frame, force, and output-folder
  validation before any recording can begin.

### v0.18.8 Beta idle residual and position-jitter repair

- Keep brief low-confidence spatial frames latched during a real contact, but
  release a residual contact after about one second of stationary full-spectrum
  evidence with no fresh spectral activity and no credible spatial fingerprint.
- Require credible spatial evidence before a slow baseline departure can arm a
  new contact. Slow drift alone can no longer reactivate the tactile surface.
- Establish a provisional visual position only around real spectral activity,
  then hold that location through quiet classifier jitter instead of moving the
  digital-twin patch between P11-P33.
- Feed the timed quiet no-contact decision into the existing baseline recovery
  guard so subsequent stable frames can re-anchor the runtime reference.
- Validate the complete source tree with 490 tests and 172 subtests.

### v0.18.7 Beta contact response repair

- Stop treating low position-classifier confidence as release evidence. A
  stationary physical press now remains latched until optical contact evidence
  falls below the contact gate, the spectrum returns near baseline, or recovery
  explicitly suppresses contact.
- Add a display-only provisional position fallback for physically verified
  contact frames. It restores the digital-twin deformation when the best
  position is useful but below the formal acceptance threshold, while keeping
  the formal position result unaccepted and visibly uncertain.
- Keep the fallback bounded by confidence and margin checks and never default
  an unknown location to P22.
- Validate the complete source tree with 488 tests and 172 subtests.

### v0.18.6 Beta live contact and position guard

- Gate learned contact output with fresh full-spectrum activity or credible
  spatial evidence, preventing stationary baseline residuals from appearing as
  an active press.
- Suppress optical force and position output whenever contact is not accepted.
- Stabilize live position with probability smoothing, confidence and margin
  checks, and a short hold instead of allowing ambiguous frames to jump across
  the array.
- Re-anchor the runtime reference after a stationary release while preserving
  the learned model and its grouped training contract.
- Validate latest-session replay at 97.4% mean active-contact recall, 99.1%
  mean active-position accuracy, and 0% false contact on the independent
  no-contact recording.
- Validate the source tree with 468 tests and 172 subtests.

### v0.18.5 Beta all-data runtime and recorded-spectrum peak mapping

- Deploy the latest optical-only all-data bundle for contact, nine-position,
  and continuous `Fz` estimation; PX6D force is supervision and validation,
  not a runtime model input.
- Drive the built-in demonstration with synchronized real 512-point BaySpec
  spectra and recorded PX6D references instead of the previous fixed-height
  synthetic spectrum.
- Discover the nine demonstration peaks automatically from the robust
  no-contact median spectrum, then track each peak inside a bounded local
  wavelength window.
- Assign the discovered peaks provisionally in ascending wavelength order as
  `P11, P12, P13, P21, P22, P23, P31, P32, P33`. This does not replace the
  independent 3x3 spatial display order or a future measured physical map.
- Suppress the preliminary 1540-1580 nm target markers when a recorded
  auto-discovered profile is active, so P11-P33 labels align with the visible
  measured peaks.
- Validate the complete source tree with 467 tests and 172 subtests.

### Baseline-referenced spectrum normalization

- Added wavelength-aligned full-spectrum normalization using
  `I(lambda,t) / I0(lambda)`, where `I0` is the accepted multi-frame
  no-contact baseline.
- Kept the original raw spectrum as the deployed recognition-model input;
  normalization is an independently selectable display and recording output.
- Extended synchronized spectrum recordings with raw counts, aligned baseline
  counts, normalized intensity ratio, method, and readiness status.
- Added explicit baseline-readiness fallback and low-reference guards. No
  per-frame min-max normalization is used.
- Kept the existing mFBG per-channel `I_i / I0_i` path separate from this
  ordinary-FBG full-spectrum normalization.

## v0.17.1 - 2026-07-31

### PX6D reconnect hardening

- Added bounded reconnect backoff for an unavailable or busy COM port instead
  of launching a new serial helper every second.
- Added explicit `port_busy_or_permission_denied` diagnostics, reconnect delay,
  and consecutive-failure status without changing the existing PX6D API.
- Reset reconnect state immediately after the first valid force frame so a
  reconnected sensor becomes available without restarting TOUCH.
- Attached isolated Windows serial helpers to a kill-on-close Job Object so a
  forced desktop shutdown cannot leave a child process holding COM3.
- Kept force filtering, tare semantics, BaySpec acquisition, recognition, and
  recording contracts unchanged.

### Validation boundary

- The complete automated suite passes with 439 tests and 170 subtests.
- Backoff behavior and Windows child-process cleanup were exercised directly.
- COM3 is currently held by a stale driver request from an earlier process, so
  live PX6D force frames still require a physical USB reconnect or Windows
  restart before hardware validation can be completed.

## v0.17.0 - 2026-07-30

### Stable 5 ms spectrum display path

- Hardened desktop port selection with a real bind check, so a stale process
  that rejects connections is not mistaken for a free backend port.
- Added auditable display-only spectrum processing with optional overlay,
  background subtraction, baseline correction, and smoothing.
- Set the default BaySpec exposure request to 5 ms and exposed 5, 10, 20, and
  40 ms choices in Settings.
- Preserved the original raw 512-point intensity array for recognition and
  synchronized recording; display processing does not alter model input.
- Added lightweight frontend source recovery so a page reload reconnects to an
  already-running SDK stream without opening a second hardware session.
- Kept the process-per-frame SDK isolation strategy because repeated in-process
  vendor SDK acquisition previously caused Windows access violations.

### Validation boundary

- Real BaySpec hardware produced 60 unique 512-point frames with nine detected
  peaks, no invalid values, and about 3.2 observed frames per second.
- The 5 ms setting is detector exposure time, not end-to-end frame period; the
  isolated helper-process startup remains the dominant latency.
- No unattended baseline was captured, so the Operator correctly remained in
  a baseline-required neutral state during the hardware check.
- Per-channel intensity normalization is intentionally deferred. A future
  implementation must use stored no-contact baselines and robust noise scales,
  never per-frame min-max normalization.

## v0.16.0 - 2026-07-29

### Acceptance remediation

- Fail closed in the Operator view for invalid, stale, fallback, or missing
  formal spectrum input while retaining raw evidence in Diagnostics.
- Keep the deployed static spectrum model as the only Operator model; the
  temporal candidate is diagnostics-only and is consumed from cache by the
  recorder.
- Require explicit no-contact attestation before baseline capture and reject
  out-of-order temporal frames.
- Make demo playback local-only, default it to a single cycle, and prevent it
  from mutating live acquisition, recording, or baseline state.
- Use one atomic presentation frame for the trace, summary, response map, and
  3D surface.
- Replace the permanent WebGL animation loop with demand-driven rendering and
  dispose GPU resources on teardown.
- Validate optical-force synchronization before recording, persist release and
  calibration provenance, journal capture progress, flush data to disk, and
  recover interrupted sessions.
- Freeze PX6D automatic zero tracking during recording and require explicit
  operator tare by default.
- Use configuration and environment overrides for BaySpec device and Sense
  export paths instead of hard-coded acquisition assumptions.
- Expose one release identity through `VERSION.json`, the health endpoint, and
  the frozen build.
- Store packaged capture output under `Documents/TOUCH/captures` by default.

### Isolated mFBG intensity path

- Added a future-primary `mfbg_intensity_3x3` profile without replacing the
  active ordinary-FBG runtime.
- Added validated nine-channel configuration, multi-frame baseline estimation,
  tracked spectral-window intensity demodulation, and coupled response frames.
- Added multi-region contact structures and explicit configured, analyzed, and
  real-enabled channel counts.
- Added tidy and wide recording adapters plus isolated
  `/api/mfbg-intensity/*` endpoints.
- Added a compact read-only sensor-profile card to Diagnostics; Operator view
  remains unchanged.
- Replaced the developer-facing runtime mode label with a concise sensor
  profile status that distinguishes active Ordinary FBG operation from the
  integrated, calibration-pending mFBG profile.
- Kept real 3x3 activation, dense reconstruction, calibrated force, and
  calibrated pressure disabled pending new real mFBG data.
- Refined the compact desktop hierarchy: whole-hand navigation is available
  immediately in Operator and fullscreen views, while Diagnostics keeps
  complete workspace labels without camera-arrow clutter.

### Validation boundary

- Automated source and contract tests cover the remediation logic.
- Real BaySpec press/release, PX6D reconnect/tare/synchronization, long-duration
  soak, power-loss interruption, Windows DPI, and clean-machine checks remain
  explicit release-validation tasks and are not claimed by this changelog.

## v0.15.4 - 2026-07-27

### Faster visible startup

- Show the native TOUCH window before importing and starting the local API.
- Use a lightweight startup view while the backend initializes, then navigate
  the same window to the complete application after its health check succeeds.
- Load Uvicorn lazily in the backend worker instead of blocking first paint.
- Add explicit startup timing logs and a readable startup failure state.

### Validation

- Python test suite passed: 263 tests and 13 subtests.
- Packaged Windows self-test completed successfully.
- Packaged startup displayed the native window in about 1.8 seconds and reached
  the full application in about 5.1 seconds on the development machine.

## v0.15.3 - 2026-07-26

### Five-finger navigation

- Corrected right navigation to follow
  `All -> Thumb -> Little -> Ring -> Middle -> Index -> Thumb -> All`.
- Kept left navigation as the reverse traversal of the same physical order.
- Replaced framed navigation buttons with translucent stationary arrows and
  reduced pressed-state movement so the controls no longer jump.

### Desktop transition and compact layout

- Added lightweight native-window snapshot transitions for minimize and restore
  without repeatedly resizing the live WebGL scene.
- Kept compact Operator and Diagnostics labels, commands, and summary content
  inside their panels at reduced window sizes.
- Preserved the trained model, BaySpec demodulation, coupling logic, PX6D
  synchronization, and five-finger geometry without behavioral changes.

### Validation

- Python test suite passed: 260 tests and 13 subtests.
- Python compilation and JavaScript syntax checks passed.
- Packaged Windows self-test completed successfully.

## v0.15.2 - 2026-07-25

### Code cleanup

- Removed unreachable legacy frontend label, formatting, spectrum-copy, and
  obsolete geometry helpers.
- Consolidated duplicated Diagnostics command-grid and header-height rules into
  one canonical layout definition.
- Kept trained recognition, BaySpec demodulation, coupling behavior, and
  five-finger geometry unchanged.

### Desktop stability

- Replaced the fixed Diagnostics header row with a content-sized row so wrapped
  commands stay inside the header at compact widths.
- Added regression coverage that prevents the removed helpers and duplicated
  critical layout declarations from returning.
- Validated the Operator and Diagnostics views at 1280 x 720 with no horizontal
  overflow or browser-console errors.

## v0.15.0 - 2026-07-24

### Five-finger tactile geometry

- Preserved the existing modified TOUCH thumb and its fitted sensor slot.
- Added provisional SolidWorks recess geometry for the index, middle, ring, and
  little fingertips.
- Enlarged each non-thumb sensing area to the available fingertip surface while
  retaining a structural border.
- Aligned each recess and sensor surface to its local fingertip tangent plane.
- Used equal-depth fitted seats so the lower edge no longer appears excessively
  recessed or steeply tilted.
- Bundled the four source STL seats, geometry manifest, integrated whole-hand
  GLB, and reproducible integration tool.

### Validation

- Four fingertip slot checks passed for depth, plane alignment, below-plane
  preservation, and removal of material above the fitted plane.
- Desktop, icon, and whole-hand integration contract tests passed: 27/27.
- The packaged whole-hand GLB SHA-256 matches the validated source asset.
- The standalone `TOUCH.exe` starts successfully with all four non-thumb sensor
  surfaces enabled.

## v0.14.0 - 2026-07-22

### Recording workflow

- Moved Data recording out of Input into a dedicated Record workspace.
- Placed Record second in Diagnostics, directly after Signal.
- Preserved synchronized spectrum, recognition, and six-axis force capture.
- Preserved the P11-P33 position selector, trial ID, notes, output folder, and
  start/stop controls.

### Diagnostics navigation

- Expanded Diagnostics from six to seven task workspaces.
- Added horizontal scrolling with readable minimum tab widths.
- Added automatic scrolling of the selected tab into view.
- Kept Input and Record mutually exclusive so acquisition evidence and capture
  controls no longer compete for vertical space.

### Runtime and UI included in this build

- New compact contact-fold application icon with a light cyan body, deep teal
  recess, and coral contact sphere; the default Python executable icon is no
  longer used.
- PX6D reconnect and conditioned six-axis reference display.
- Synchronized optical-force data recording for later force calibration.
- Compact Operator evidence layout and corrected fullscreen summary behavior.
- Existing trained static-spectrum recognition runtime and demo behavior.

### Validation

- JavaScript syntax check passed.
- 123 Python tests passed.
- Local browser validation confirmed Record/Input separation, real horizontal
  tab overflow, automatic tab scrolling, and no console errors.
