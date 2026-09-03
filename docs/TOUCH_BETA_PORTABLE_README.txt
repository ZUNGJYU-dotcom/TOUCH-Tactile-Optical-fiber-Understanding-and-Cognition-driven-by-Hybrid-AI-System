TOUCH Beta portable release
===========================

This package contains the current TOUCH Beta application as one Windows EXE.
Python, the trained runtime model, the web interface, runtime configuration,
the BaySpec x86 acquisition helper, and the vendor SDK DLL are embedded.

Install
-------
Run the Setup EXE, or copy the standalone EXE to a writable local folder and
start it directly. The installer uses the current Windows account and writes to:

  %LOCALAPPDATA%\Programs\TOUCH Beta

Captured data is kept outside the application at:

  %USERPROFILE%\Documents\TOUCH\captures

Hardware requirements
---------------------
The application package includes the user-mode BaySpec helper and SDK DLL. A
target computer still needs the hardware vendor's matching Windows USB driver.
PX6D use also requires its Windows serial/USB driver. Drivers are system-level
components and are not silently installed by this package.

Windows 10/11 x64 and Microsoft Edge WebView2 Runtime are required. WebView2 is
normally present on supported Windows systems.

Validation
----------
Before connecting hardware, run this from PowerShell or Command Prompt:

  TOUCH-Beta-v0.19.17.exe --self-test

A successful self-test returns exit code 0. The first start of the one-file
edition can take longer because Windows extracts its embedded runtime to a
private temporary directory.

Default BaySpec acquisition
---------------------------
  Sensor mode: High Sensitivity
  Exposure:    300 us
  Poll target: 10 ms

Record pipeline
---------------
Raw 512-point spectrum and PX6D Fz capture is independent of model inference.
Diagnostics reports the measured capture rate. Recognition rows use only an
exact same-frame cached prediction; deferred rows can be reconstructed offline
from the retained raw spectrum.

Release channel: Beta. Do not use this package to replace a validated Stable
installation without a separate promotion decision.
