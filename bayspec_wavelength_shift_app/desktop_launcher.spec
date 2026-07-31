# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules


app_dir = Path.cwd()
datas = [
    (str(app_dir / "frontend"), "frontend"),
    (str(app_dir / "sdk_probe"), "sdk_probe"),
    (str(app_dir.parent / "VERSION.json"), "."),
    (str(app_dir.parent / "config"), "config"),
    (str(app_dir.parent / "src"), "src"),
    (str(app_dir.parent / "models"), "models"),
]

hiddenimports = (
    collect_submodules("backend")
    + collect_submodules("src.array_surface")
    + collect_submodules("src.mfbg_intensity")
    + collect_submodules("src.wavelength_shift")
    + collect_submodules("src.hybrid_spectrum")
    + collect_submodules("uvicorn")
    + collect_submodules("websockets")
    + collect_submodules("serial")
    + [
        "clr",
        "System",
        "sklearn.pipeline",
        "sklearn.impute._base",
        "sklearn.preprocessing._data",
        "sklearn.linear_model._logistic",
        "sklearn.ensemble._forest",
        "sklearn.cross_decomposition",
        "sklearn.cross_decomposition._pls",
    ]
)

# These packages are present in the research environment but are not used by
# the deployed static-spectrum runtime. Excluding them prevents optional
# sklearn compatibility hooks from adding several hundred megabytes.
deployment_excludes = [
    "torch",
    "tensorflow",
    "sympy",
    "matplotlib",
    "pandas",
]

a = Analysis(
    ["desktop_launcher.py"],
    pathex=[str(app_dir), str(app_dir.parent)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=deployment_excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TOUCH",
    icon=str(app_dir / "assets" / "touch_system_icon.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TOUCH",
)
