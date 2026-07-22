# Changelog

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
