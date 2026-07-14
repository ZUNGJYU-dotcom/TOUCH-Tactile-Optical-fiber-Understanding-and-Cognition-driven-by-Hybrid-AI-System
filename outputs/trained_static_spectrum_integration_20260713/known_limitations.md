# Known Limitations

- The formal model evidence is from one acquisition session with leave-one-repeat-index-out evaluation.
- Cross-session, day-to-day, remounting, temperature, and different-operator generalization have not yet been established.
- Position means an approximate manual fingertip contact region, not a calibrated contact centroid.
- Light, normal, and hard are approximate manual response levels, not force_N or calibrated pressure.
- The push-pull gauge data uses a much smaller contact tip and remains a separate reference domain.
- Force-level features remain more baseline-sensitive than the current-spectrum position features.
- A stable post-release recovery baseline is required before inference.
- The current data contains static snapshots only; tap, slide, and other temporal gestures are outside this model.
- The current P11-P33 labels are supervised recognition classes. This does not enable an independently calibrated real 3x3 force field.
- The BaySpec/Sense or direct-SDK acquisition path still needs to provide valid 512-point spectra at runtime.
