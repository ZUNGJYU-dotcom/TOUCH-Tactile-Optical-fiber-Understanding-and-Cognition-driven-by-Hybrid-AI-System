# Known limitations and next collection plan

- All spectra were collected in one acquisition session; session drift may remain correlated with labels.
- Each manual position-force condition has five independent snapshot files.
- Manual `light/normal/hard` labels are approximate operator levels, not calibrated force_N.
- Manual contact covers a larger, approximate fingertip area; gauge contact uses a small tip at an exact FBG point, so the two domains are intentionally not pooled.
- Baseline normalization uses two time-local no-contact clusters and interpolation between them.
- Those no-contact spectra were generally captured after a press was released and the sensor recovered; they are not ideal cold-start references.
- The contact detector has only 15 no-contact files and should be strengthened with interleaved baselines.
- For the next session, randomize position and force order and capture both a cold-start baseline and a settled post-release recovery baseline every 3-5 trials.
- Capture at least 20-30 independent spectra per position-force condition across at least three sessions before training CNNs.
- Capture continuous sequences separately if tap and slide recognition are needed.
