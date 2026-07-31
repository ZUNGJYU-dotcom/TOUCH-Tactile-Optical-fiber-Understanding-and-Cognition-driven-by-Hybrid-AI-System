# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules


app_dir = Path.cwd()
project_dir = app_dir.parent
latest_model = (
    project_dir
    / "models"
    / "candidates"
    / "ordinary_fbg_optical_only_force_candidate.joblib"
)

# Beta intentionally contains one deployable model. The source repository keeps
# historical models for Stable and research comparison, but none of them are
# copied into this package.
datas = [
    (str(app_dir / "frontend"), "frontend"),
    (str(app_dir / "sdk_probe"), "sdk_probe"),
    (str(app_dir / "assets" / "demo"), "assets/demo"),
    (str(app_dir / "beta_all_data_runtime.flag"), "."),
    (str(project_dir / "VERSION.json"), "."),
    (str(project_dir / "config"), "config"),
    (str(project_dir / "src"), "src"),
    (str(latest_model), "models/candidates"),
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

deployment_excludes = [
    "torch",
    "tensorflow",
    "sympy",
    "matplotlib",
    "pandas",
]

a = Analysis(
    ["desktop_launcher.py"],
    pathex=[str(app_dir), str(project_dir)],
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
