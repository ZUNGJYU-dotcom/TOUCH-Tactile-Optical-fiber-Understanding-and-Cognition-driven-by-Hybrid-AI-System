# Edition Boundaries

## This Edition

**TOUCH System - BaySpec FBG Wavelength-Shift Demodulation Edition**

- Input: BaySpec wavelength-calibrated spectrum.
- Baseline: synchronized no-contact baseline for `FBG01` through `FBG09`.
- Formal input: joint wavelength, intensity, area, and spectral-shape fingerprint.
- Diagnostic response: signed `delta lambda = lambdaB - lambda0`, in pm.
- Current real mode: global nine-FBG spectrum with physical P11-P33 mapping pending.
- P22 role: legacy full-spectrum transport and single-point diagnostic fallback only.
- Spatial array mode: simulated scaffold only until labelled mapping is approved.

## Separate Preserved Editions

**PD Voltage Edition** uses TiePie oscilloscope voltage and `V / V0`.

**Optical Intensity Edition** uses BaySpec intensity and `I / I0` attenuation.

Neither edition is included, overwritten, renamed, or launched by this project.
The wavelength-shift app uses port 8630 so the source trees and runtime
identities remain distinct.

## Claims

This edition may report Bragg wavelength displacement, peak-tracking quality,
and normalized uncalibrated response. It does not claim calibrated strain,
temperature, displacement, pressure, force, `force_N`, or measured 3x3
reconstruction.
