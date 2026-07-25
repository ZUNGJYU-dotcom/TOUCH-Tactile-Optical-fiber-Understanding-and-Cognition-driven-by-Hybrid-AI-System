# Changelog

## v0.15.2 - 2026-07-25

### Code cleanup

- Removed unreachable legacy frontend label, formatting, spectrum-copy, and
  obsolete geometry helpers.
- Consolidated duplicated Diagnostics command-grid and header-height rules into
  one canonical layout definition.
- Kept trained recognition, BaySpec demodulation, coupling behavior, and
  five-finger geometry unchanged.

### Desktop stability

- Replaced the fixed Diagnostics header row with a content-sized row so wrapped
  commands stay inside the header at compact widths.
- Added regression coverage that prevents the removed helpers and duplicated
  critical layout declarations from returning.
- Validated the Operator and Diagnostics views at 1280 x 720 with no horizontal
  overflow or browser-console errors.

## v0.15.0 - 2026-07-24

### Five-finger tactile geometry

- Preserved the existing modified TOUCH thumb and its fitted sensor slot.
- Added provisional SolidWorks recess geometry for the index, middle, ring, and
  little fingertips.
- Enlarged each non-thumb sensing area to the available fingertip surface while
  retaining a structural border.
- Aligned each recess and sensor surface to its local fingertip tangent plane.
- Used equal-depth fitted seats so the lower edge no longer appears excessively
  recessed or steeply tilted.
- Bundled the four source STL seats, geometry manifest, integrated whole-hand
  GLB, and reproducible integration tool.

### Validation

- Four fingertip slot checks passed for depth, plane alignment, below-plane
  preservation, and removal of material above the fitted plane.
- Desktop, icon, and whole-hand integration contract tests passed: 27/27.
- The packaged whole-hand GLB SHA-256 matches the validated source asset.
- The standalone `TOUCH.exe` starts successfully with all four non-thumb sensor
  surfaces enabled.

## v0.14.0 - 2026-07-22

### Recording workflow

- Moved Data recording out of Input into a dedicated Record workspace.
- Placed Record second in Diagnostics, directly after Signal.
- Preserved synchronized spectrum, recognition, and six-axis force capture.
- Preserved the P11-P33 position selector, trial ID, notes, output folder, and
  start/stop controls.

### Diagnostics navigation

- Expanded Diagnostics from six to seven task workspaces.
- Added horizontal scrolling with readable minimum tab widths.
- Added automatic scrolling of the selected tab into view.
- Kept Input and Record mutually exclusive so acquisition evidence and capture
  controls no longer compete for vertical space.

### Runtime and UI included in this build

- New compact contact-fold application icon with a light cyan body, deep teal
  recess, and coral contact sphere; the default Python executable icon is no
  longer used.
- PX6D reconnect and conditioned six-axis reference display.
- Synchronized optical-force data recording for later force calibration.
- Compact Operator evidence layout and corrected fullscreen summary behavior.
- Existing trained static-spectrum recognition runtime and demo behavior.

### Validation

- JavaScript syntax check passed.
- 123 Python tests passed.
- Local browser validation confirmed Record/Input separation, real horizontal
  tab overflow, automatic tab scrolling, and no console errors.
