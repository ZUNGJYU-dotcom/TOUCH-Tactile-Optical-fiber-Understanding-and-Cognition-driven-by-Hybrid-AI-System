# Trained Static Spectrum Model Summary

## Scope

This is the independent ordinary-FBG BaySpec full-spectrum recognition edition. It does not replace or modify the retained PD-voltage and optical-intensity editions.

The primary deployment domain is broad manual fingertip contact. The outputs are:

- contact versus no contact;
- approximate contact region P11-P33;
- approximate response level: light, normal, or hard;
- a position-linked deformation proxy for the digital twin.

The response levels are not calibrated force values and the software does not output force_N.

## Training data

- 240 independent static 512-point spectrum CSV snapshots;
- 15 post-press/release recovery no-contact spectra;
- 135 manual fingertip spectra: 9 positions x 3 levels x 5 repeats;
- 90 push-pull gauge spectra kept as a separate reference domain;
- no temporal tap or slide sequence is present in this dataset.

Manual and push-pull gauge records are not pooled because their contact area and mechanics are different.

## Selected models

| Task | Model | Feature view | Grouped OOF accuracy | Grouped OOF macro-F1 |
| --- | --- | --- | ---: | ---: |
| Contact detection | LogisticRegression | current spectrum shape | 1.0000 | 1.0000 |
| Approximate position | ExtraTrees | current spectrum shape | 1.0000 | 1.0000 |
| Manual response level | ExtraTrees | engineered baseline-relative features | 0.9778 | 0.9778 |
| Position-conditioned response level | ExtraTrees | engineered baseline-relative features | 0.9778 | 0.9778 |

The hierarchical joint position-and-level accuracy is 0.9778 and the joint 27-class macro-F1 is 0.9777 under leave-one-repeat-index-out evaluation.

These values are a single-session baseline, not a final cross-session generalization claim.

## Runtime decision path

1. Acquire a stable multi-frame post-release recovery baseline.
2. Use current-spectrum shape features for contact and position.
3. Use position-conditioned baseline-relative features for light, normal, and hard.
4. Map P11-P33 to the physical digital-twin coordinates.
5. Suppress deformation when the contact detector returns no_contact.

The packaged EXE has been checked with P13-light, P22-hard, P31-normal, and no-contact fixtures.
