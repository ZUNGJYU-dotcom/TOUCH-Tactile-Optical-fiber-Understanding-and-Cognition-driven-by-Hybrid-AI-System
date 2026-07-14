# Runtime Recovery Baseline Contract

The available no-contact records were captured mainly after a press and subsequent release. They are therefore treated as `post_press_release_recovery_no_contact`, not as ideal cold-start baselines.

Before trained inference is enabled, the runtime must collect at least 20 full-spectrum frames spanning at least 0.6 seconds. The bridge computes a median spectrum and checks normalized frame noise and drift. Inference remains blocked when the recovery baseline is too short or unstable.

Accepted runtime states are:

- `stable_post_release_recovery_baseline`;
- `stable_post_release_recovery_baseline_with_warning` when the response remains usable but baseline quality deserves attention.

Rejected states include:

- `insufficient_recovery_baseline_frames`;
- `unstable_post_release_recovery_baseline`.

Operator procedure:

1. Release the sensor completely.
2. Wait for the spectrum to recover and become visually stable.
3. Set baseline while no contact is present.
4. Begin position/level inference only after the software reports baseline ready.
5. Re-baseline after hardware movement, source drift, sensor remounting, or persistent residual response.

The next collection campaign should record both cold-start no-contact baselines and post-release recovery baselines in multiple sessions.
