# Portable Release

## Stable v0.19.25 one-file release

The formal transferable application is:

`TOUCH-Stable-v0.19.25-Windows-x64.exe`

- Size: 103,285,538 bytes (98.50 MiB)
- SHA-256: `7963470D59CDA9541EDC7120F66B342D2A451190EB2EE0AC0182D8B139B4EF94`

The single EXE embeds the current hash-bound model, deployment metadata, Python
runtime, frontend and 3D assets, Stable runtime configuration, BaySpec x86
acquisition helper, and the vendor user-mode SDK DLL. It does not require a
neighboring `_internal` folder or a Python installation.

Before connecting hardware, run:

```powershell
& '.\TOUCH-Stable-v0.19.25-Windows-x64.exe' --self-test
```

The self-test validates the embedded frontend, backend contract, SDK helper,
runtime model, mFBG configuration, and release identity without opening
hardware or binding port `8640`. A successful self-test returns exit code `0`.
Details are written to `%LOCALAPPDATA%\TOUCH\logs\desktop_launcher.log`.

The first launch can be slower than the folder-based build because PyInstaller
extracts the embedded runtime to a private temporary directory. Verify the
published SHA-256 value before transfer or installation.

## Target-computer hardware requirements

The BaySpec helper and vendor SDK DLL are embedded, but a matching Windows USB
kernel driver must still be installed on the target computer. PX6D use also
requires its serial/USB driver. System drivers are not silently installed by
the application package.

Windows 10/11 x64 and Microsoft Edge WebView2 Runtime are required. WebView2 is
normally present on supported Windows systems.

## Release boundary

The one-file package contains the complete TOUCH user-mode application but does
not install kernel drivers. Stable remains the ordinary-FBG BaySpec edition;
the separate mFBG research Beta must retain independent models, configuration,
ports, release directories, and shortcuts.
