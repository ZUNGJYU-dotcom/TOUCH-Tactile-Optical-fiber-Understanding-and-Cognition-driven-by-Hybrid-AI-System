# Ordinary FBG static spectrum baseline training

## Scope

- Input: one 512-point BaySpec/Sense full-spectrum snapshot per CSV.
- Primary outputs: contact state, P11-P33 contact position, and approximate manual force level.
- Manual labels map to `light`, `normal`, and `hard` exactly as supplied by the operator.
- Manual pressing is an approximate-position, broad fingertip contact patch and is the primary deployment domain.
- Gauge pressing is a precise FBG-point load with a much smaller tip; it is evaluated only as a separate reference domain and is not merged into the primary models.
- Gauge labels `0.1N` and `0.2N` are not merged with manual levels.
- The available `no_contact` files are post-press release/recovery states rather than ideal cold-start baselines.
- Runtime inference therefore requires a stable multi-frame baseline captured after release and settling.
- No random snapshot split is used. Every metric is leave-one-repeat-index-out over groups 1-5.
- Raw CSV files were read only and remain immutable.

## Dataset

- Total independent spectrum files: **240**.
- No-contact/manual/gauge files: **{'manual_press': 135, 'gauge_press': 90, 'no_contact': 15}**.
- Baseline clusters: **2**, sizes [10, 5].
- Device temperature range: **32.46-33.16 C**.
- Source QA flags: **none**.

## Selected deployable-shape baselines

| Task | Model | Features | OOF accuracy | OOF macro-F1 |
|---|---|---|---:|---:|
| contact_detector | logistic_regression | current_shape | 1.0000 | 1.0000 |
| position_classifier | extra_trees | current_shape | 1.0000 | 1.0000 |
| manual_force_classifier | extra_trees | engineered | 0.9778 | 0.9778 |
| position_conditioned_force_classifier | extra_trees | engineered | 0.9778 | 0.9778 |

## End-to-end manual snapshot check

- Position and force both correct: **0.9778**.
- Joint 27-class macro-F1: **0.9777**.

## Interpretation

The saved bundle is shaped for the digital twin: no-contact suppresses deformation, the manual-domain position classifier supplies the response center, and the position-conditioned force classifier supplies a light/normal/hard deformation proxy.
A global manual force model is retained only as a diagnostic fallback. The primary force decision uses one model per predicted P11-P33 position because the spectral force signature is position dependent.
This is a single-session baseline, not a final generalization claim. A second acquisition session with randomized condition order is required before live deployment is considered reliable.
A CNN is intentionally deferred: 240 static snapshots and only five repeats per condition are insufficient for a defensible temporal or full-spectrum deep model.
These CSVs do not contain temporal sequences, so they cannot train tap, slide direction, or release dynamics.
The position model uses baseline-independent current-shape features and remained stable across the early and late recovery fixtures. The force-level model still uses baseline-relative features, so a current-session stable recovery baseline is mandatory.
