# Algorithm Reference

The implementation follows standard FBG interrogation practice: track the
spectral feature's wavelength relative to a no-contact reference rather than
using peak intensity as the measurand.

## Implemented Estimators

1. Baseline-spectrum normalized cross correlation for robust translation
   tracking.
2. Intensity-weighted centroid in the channel search window as fallback.
3. Three-point parabolic interpolation around the extremum for sub-pixel
   checking.
4. Max-point wavelength retained only as a debug marker.

## Quality Checks

- minimum cross-correlation coefficient;
- centroid/parabolic disagreement;
- frame-to-frame wavelength jump;
- peak SNR, saturation, dark signal, and search-window edge;
- baseline wavelength noise and noise-gated no-contact threshold.

## Interpretation

`delta lambda` contains both strain and temperature effects. Converting it to
strain requires a sensor sensitivity model and temperature compensation.
Converting it to force additionally requires mechanical calibration of the
sensor package. Neither conversion is performed in this edition.

## Primary References

- BaySpec FBG Interrogation Analyzer product documentation and WaveCapture
  FBGA datasheet.
- Peer-reviewed FBG demodulation literature on centroid, local fitting, and
  cross-correlation peak tracking.
