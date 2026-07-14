# FBG Wavelength-Shift Edition Pass 1

- Forked the mature BaySpec acquisition and TOUCH UI into an independent
  source tree without modifying the optical-intensity edition.
- Added configurable Bragg peak tracking with baseline-spectrum cross
  correlation, weighted-centroid fallback, and parabolic-fit checking.
- Added frozen no-contact `lambda0`, signed `delta lambda` in pm, red/blue
  direction, baseline noise gating, correlation QA, estimator disagreement,
  and frame-jump QA.
- Converted the trace, summary, diagnostics, spectrum markers, response bands,
  and 3D proxy to wavelength-shift semantics.
- Converted simulated 9-FBG spectra to move peak positions while keeping peak
  intensity nearly constant.
- Removed the optical-intensity cascade from the active wavelength simulation;
  simulated coupling is contact-area plus shared-elastomer mechanical response.
- Assigned independent runtime identity and port 8630.
- Added 13 wavelength-demodulation and array regression tests.
- Fixed the wavelength-demo spectrum contract: every FBG peak keeps a fixed
  per-channel height and width, only its center wavelength moves, and each
  spectrum frame replaces the prior frame without per-bin intensity blending.

This pass does not enable measured 3x3 reconstruction or calibrated force.
