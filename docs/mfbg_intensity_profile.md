# mFBG 3x3 Optical-Intensity Profile

## Purpose

This profile prepares TOUCH for the future Micro-FBG research line without
changing the retained ordinary-FBG runtime. The current packaged application
continues to use the ordinary-FBG hybrid spectral model. The new profile is an
isolated acquisition, demodulation, and recording path for later mFBG data.

Current profile boundary:

- active runtime: `ordinary_fbg_hybrid_spectral`;
- future primary profile: `mfbg_intensity_3x3`;
- primary mFBG signal: spectral-window optical intensity;
- real 3x3 activation: disabled;
- calibrated force or pressure output: unavailable;
- dense non-grating reconstruction: interface only, pending real training data.

## Physical Channel Layout

The display and spatial order is:

```text
P11  P21  P31
P12  P22  P32
P13  P23  P33
```

The three fibers are configured as:

```text
fiber 1: P13 -> P12 -> P11
fiber 2: P23 -> P22 -> P21
fiber 3: P33 -> P32 -> P31
```

These paths describe expected downstream optical coupling. They are metadata,
not a hard-coded inverse model. Cross-fiber mechanical coupling is also
retained as metadata until measured calibration data are available.

## Intensity Demodulation

For each FBG channel, the demodulator:

1. selects a search window around the measured wavelength, or the preliminary
   target wavelength when no measured value exists;
2. estimates a local peak centroid;
3. limits implausible frame-to-frame tracking displacement;
4. integrates a fixed-width spectral window around the tracked center;
5. divides the integral by its wavelength span to obtain a grid-stable mean
   intensity value;
6. compares that value with a multi-frame median baseline.

The primary output metrics are:

```text
relative_intensity = I / I0
attenuation_ratio = 1 - I / I0
loss_db = -10 * log10(I / I0)
```

An intensity rise is a QA warning and cannot be converted into a false contact
response. Wavelength shift remains available as a diagnostic quantity but is
not the primary mFBG response metric.

## Frame Contract

Every analyzed spectrum returns:

- nine ordered channel records and a `channel_map`;
- `intensity_vector`, `baseline_intensity_vector`,
  `relative_intensity_vector`, `attenuation_vector`, and `loss_db_vector`;
- responding channel IDs and count;
- one or more connected optical-response regions;
- a continuous coupled-response surface proxy;
- explicit configured, analyzed, and real-enabled channel counts;
- QA state and activation state;
- explicit `real_3x3_enabled`, `force_N_output`, and calibrated-output flags.

Multiple separated response regions remain separate in the frame contract.
The pipeline does not force every frame into one dominant contact point.

## Baseline and Recording

A baseline requires at least 20 full-spectrum frames. Each channel baseline is
the median intensity, with MAD-based noise recorded independently.

Recording adapters provide:

- one tidy row per channel per frame;
- one wide synchronized row per spectrum frame;
- optional retained raw spectrum;
- compatibility with a separate force-sensor reference stream.

Raw spectra should remain immutable. Future labels and reconstructed fields
should be stored as derived data with source frame IDs.

## API

The isolated endpoints are:

```text
GET  /api/mfbg-intensity/profiles
GET  /api/mfbg-intensity/profile
GET  /api/mfbg-intensity/frame
POST /api/mfbg-intensity/analyze-spectrum
POST /api/mfbg-intensity/analyze-latest-bayspec-frame
POST /api/mfbg-intensity/baseline
POST /api/mfbg-intensity/baseline-from-recent-bayspec-frames
POST /api/mfbg-intensity/reset
GET  /api/mfbg-intensity/recording-preview
```

The ordinary-FBG `/api/frame` and trained spectrum endpoints are unchanged.

## Activation Gate

Real mFBG 3x3 mode must remain disabled until all of the following exist:

1. a clean no-contact spectrum from the fabricated array;
2. measured center wavelengths for all usable channels;
3. per-channel baseline and noise validation;
4. confirmed channel order and physical orientation;
5. real contact data for coupling characterization;
6. independent validation of any spatial reconstruction model.

The preliminary 1540-1580 nm wavelength table is only a target plan. It must
not be presented as measured hardware configuration.

## Deferred Work

The following are intentionally not implemented as measured results yet:

- learned multi-contact recognition;
- dense localization between FBG points;
- shape reconstruction;
- coupling inversion;
- optical force regression;
- real five-finger mFBG deployment.

The interfaces are ready for those stages, but models must be trained only
after new real mFBG data are collected with grouped experiment splits.
