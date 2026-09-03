# TOUCH Stable v0.19.25

Release date: 2026-09-03

## Download

`TOUCH-Stable-v0.19.25-Windows-x64.exe`

- Size: 103,285,538 bytes (98.50 MiB)
- SHA-256: `7963470D59CDA9541EDC7120F66B342D2A451190EB2EE0AC0182D8B139B4EF94`

This is a true Windows x64 single-file build. It embeds the current Stable
model, hash-bound deployment metadata, Python runtime, frontend and 3D assets,
Stable configuration, BaySpec x86 acquisition helper, and vendor user-mode SDK
DLL. It does not require Python or a neighboring `_internal` directory.

## Stable Runtime

- BaySpec High Sensitivity is the default mode with 300 us integration.
- Contact uses LightGBM over complete baseline-relative spectral evidence and
  the joint nine-FBG fingerprint.
- Position uses a random forest over the complete joint nine-FBG fingerprint.
- Optical `Fz` uses histogram gradient boosting and has no display-side upper
  clip; approximately 0-6.98 N is the observed training range.
- Runtime inference is optical-only. PX6D supplies training supervision and
  Diagnostics reference data, but is not a model input.
- Exactly one current model is packaged; legacy and Beta model fallbacks are
  excluded from the Stable runtime.

The model contains 49,853 frames from 161 same-day High Sensitivity / 300 us
sessions. Grouped out-of-file evaluation produced 96.29% contact macro-F1,
98.06% position accuracy, 98.05% position macro-F1, 0.310 N force MAE, and
0.929 force R2. Blind1, Blind3, and Blind4 became answer-known development data
during model selection, so a new-date independent validation remains required.

## Runtime Repairs

- Late concurrent SDK frames can no longer rewind temporal state.
- Contact release can recover from a saturated classifier only after the full
  nine-FBG fingerprint returns toward the armed baseline under conservative
  quiet-state conditions.
- Stable owns independent acquisition and contact-state settings, preventing
  old Stable preferences or later Beta experiments from changing its behavior.
- Record retains synchronized raw spectra, runtime response, and PX6D reference
  force with explicit same-frame or deferred-response provenance.

## Verification

Run before connecting hardware:

```powershell
& '.\TOUCH-Stable-v0.19.25-Windows-x64.exe' --self-test
```

The release artifact passed the frozen self-test, including frontend, backend
contract, embedded SDK helper, current model loading, and the single-model
isolation contract. The source test suite is also run before publication.

## Target Computer

Windows 10/11 x64 and Microsoft Edge WebView2 Runtime are required. The BaySpec
helper and user-mode SDK DLL are inside the EXE, but Windows kernel drivers
cannot be silently embedded and installed by a portable application. Install
the matching BaySpec USB driver and, when PX6D is used, its serial/USB driver on
the target computer.

This Stable package is the ordinary-FBG BaySpec edition. The separate mFBG
intensity research Beta retains isolated models, configuration, ports, release
directories, and shortcuts.
