# Portable Release

The Windows build is `TOUCH\TOUCH.exe` and uses local port `8640`. The release
Beta archive is `TOUCH-v0.18.5-beta-windows-x64.zip`.

Keep the executable beside its `_internal` directory. Live acquisition also
requires the matching BaySpec driver and vendor SDK/runtime on the target
computer. Captures default to `%USERPROFILE%\Documents\TOUCH\captures`, outside
the replaceable application folder.

The archive must contain the same `VERSION.json` identity exposed by
`GET /api/health`. It must not contain or replace the separate PD-voltage or
earlier optical-intensity editions.

Before hardware use, run:

```powershell
.\TOUCH\TOUCH.exe --self-test
```

The self-test validates bundled frontend, backend contract, SDK helper, model,
mFBG configuration, and release resources without starting acquisition.

Version 0.17.1 retains the 5 ms BaySpec display path and adds bounded PX6D
reconnect backoff plus Windows child-process cleanup. Recognition and recording
still consume the raw 512-point spectrum. Live PX6D validation requires COM3
to be released by the CH343 driver on the target machine.
