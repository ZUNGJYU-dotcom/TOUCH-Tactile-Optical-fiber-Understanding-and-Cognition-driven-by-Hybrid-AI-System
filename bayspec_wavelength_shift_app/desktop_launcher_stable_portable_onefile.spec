# -*- mode: python ; coding: utf-8 -*-

import json
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules


app_dir = Path.cwd()
project_dir = app_dir.parent
deployed_model = (
    project_dir
    / "models"
    / "deployed"
    / "ordinary_fbg_current_runtime.joblib"
)
deployment_metadata = deployed_model.with_suffix(".deployment.json")
stable_version = app_dir / "release_manifests" / "stable" / "VERSION.json"
version_payload = json.loads(stable_version.read_text(encoding="utf-8"))
numeric_version = str(version_payload["version"]).split("-", 1)[0]
runtime_config_names = (
    "bayspec_wavelength_shift_channels.yaml",
    "hybrid_spectrum_channels.yaml",
    "runtime_contact_state.yaml",
    "runtime_contact_state_stable.yaml",
    "px6d_reference.yaml",
    "measurement_analysis.yaml",
    "mfbg_intensity_3x3.yaml",
    "spectrum_processing.yaml",
    "thumb_holder_scene.yaml",
)

# PyInstaller extracts these embedded resources into its private runtime
# directory. The launcher resolves that directory through sys._MEIPASS.
datas = [
    (str(app_dir / "frontend"), "frontend"),
    (str(app_dir / "sdk_probe"), "sdk_probe"),
    (str(app_dir / "assets" / "demo"), "assets/demo"),
    (str(stable_version), "."),
    (str(deployed_model), "models/deployed"),
    (str(deployment_metadata), "models/deployed"),
    *[
        (str(project_dir / "config" / config_name), "config")
        for config_name in runtime_config_names
    ],
]

hiddenimports = (
    collect_submodules("backend")
    + collect_submodules("uvicorn")
    + collect_submodules("websockets")
    + collect_submodules("serial")
    + collect_submodules("lightgbm")
    + collect_submodules("sklearn.ensemble._hist_gradient_boosting")
    + [
        "clr",
        "System",
        "sklearn.pipeline",
        "sklearn.impute._base",
        "sklearn.preprocessing._data",
        "sklearn.linear_model._logistic",
        "sklearn.linear_model._ridge",
        "sklearn.ensemble._forest",
        "sklearn.cross_decomposition",
        "sklearn.cross_decomposition._pls",
        "src.array_surface.surface_mapper",
        "src.mfbg_intensity.config",
        "src.mfbg_intensity.demodulator",
        "src.mfbg_intensity.profiles",
        "src.mfbg_intensity.recording",
        "src.wavelength_shift.demodulator",
        "src.hybrid_spectrum.all_source_runtime_adapter",
        "src.hybrid_spectrum.baseline_relative_features",
        "src.hybrid_spectrum.dataset",
        "src.hybrid_spectrum.features",
        "src.hybrid_spectrum.joint_nine_fbg_features",
        "src.hybrid_spectrum.tracking",
        "src.hybrid_spectrum.runtime_baseline_guard",
        "src.hybrid_spectrum.runtime_channel_response",
        "src.hybrid_spectrum.runtime_spectral_features",
        "src.hybrid_spectrum.runtime_temporal_features",
        "src.hybrid_spectrum.runtime_literature_features",
        "src.hybrid_spectrum.measurement_consistency",
        "src.hybrid_spectrum.measurement_estimate_sources",
        "src.hybrid_spectrum.sense_fast_dat",
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
    binaries=collect_dynamic_libs("lightgbm"),
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

# Passing binaries and data directly to EXE produces one transferable file.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=f"TOUCH-Stable-v{numeric_version}-Windows-x64",
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
    runtime_tmpdir=None,
)
