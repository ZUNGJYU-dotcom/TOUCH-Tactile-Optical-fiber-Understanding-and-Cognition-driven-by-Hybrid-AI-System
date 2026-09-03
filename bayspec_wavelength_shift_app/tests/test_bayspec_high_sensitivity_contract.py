from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sdk_live import (
    DEFAULT_INTEGRATION_US,
    DEFAULT_SENSOR_MODE,
    SENSOR_MODE_HIGH_SENSITIVITY,
    BaySpecSdkLiveReader,
    normalize_integration_us,
    normalize_sensor_mode,
)
from spectrum_processing import SpectrumDisplayProcessor


APP_ROOT = Path(__file__).resolve().parents[1]


class _BridgeStub:
    pass


class _CompletedStreamProcess:
    def __init__(self, stdout: str) -> None:
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO("")
        self.returncode = 0
        self.pid = 1234

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9

    def terminate(self) -> None:
        self.returncode = -15


def test_beta_defaults_to_high_sensitivity_at_300_microseconds(tmp_path: Path) -> None:
    reader = BaySpecSdkLiveReader(bridge=_BridgeStub(), app_root=tmp_path)

    assert DEFAULT_SENSOR_MODE == SENSOR_MODE_HIGH_SENSITIVITY == 0
    assert DEFAULT_INTEGRATION_US == 300
    assert reader.status()["sensor_mode"] == 0
    assert reader.status()["sensor_mode_label"] == "High Sensitivity"
    assert reader.status()["integration"] == 300
    assert reader._helper_command(  # noqa: SLF001 - acquisition contract test.
        integration=300,
        sensor_mode=0,
    ) == [
        str(reader.helper_path),
        "--interval-ms",
        "10",
        "--integration",
        "300",
        "--sensor-mode",
        "0",
        "--frames",
        "0",
    ]
    assert reader.status()["acquisition_strategy"] == (
        "persistent_sdk_stream_with_restart"
    )


def test_beta_uses_an_isolated_settings_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("TOUCH_RELEASE_CHANNEL", "beta")

    processor = SpectrumDisplayProcessor()

    assert processor.user_settings_path == (
        tmp_path / "TOUCH" / "spectrum_processing_beta.json"
    )
    assert processor.status()["settings"]["integration_us"] == 300
    assert processor.status()["settings"]["sensor_mode"] == 0


def test_stable_uses_an_isolated_high_sensitivity_settings_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("TOUCH_RELEASE_CHANNEL", "stable")

    processor = SpectrumDisplayProcessor()

    assert processor.user_settings_path == (
        tmp_path / "TOUCH" / "spectrum_processing_stable.json"
    )
    assert processor.status()["settings"]["integration_us"] == 300
    assert processor.status()["settings"]["sensor_mode"] == 0


def test_persistent_stream_accepts_multiple_frames_without_process_restart(
    tmp_path: Path,
) -> None:
    reader = BaySpecSdkLiveReader(bridge=_BridgeStub(), app_root=tmp_path)
    reader.desired_active = True
    reader.generation = 1
    messages = [
        {
            "type": "status",
            "sensor_mode": 0,
            "integration": 300,
        },
        {"type": "spectrum", "counts": [100.0] * 512},
        {"type": "spectrum", "counts": [101.0] * 512},
    ]
    process = _CompletedStreamProcess(
        "".join(json.dumps(message) + "\n" for message in messages)
    )
    spectrum_count = 0

    def handle_message(message: dict, *, generation: int | None = None) -> bool:
        nonlocal spectrum_count
        if message.get("type") != "spectrum":
            return False
        spectrum_count += 1
        if spectrum_count == 2:
            reader.desired_active = False
            reader._stop_event.set()
        return True

    with patch("sdk_live.subprocess.Popen", return_value=process) as popen, patch.object(
        reader,
        "_handle_message",
        side_effect=handle_message,
    ):
        reader._persistent_supervisor_loop(generation=1)  # noqa: SLF001

    assert popen.call_count == 1
    assert popen.call_args.args[0][-2:] == ["--frames", "0"]
    assert spectrum_count == 2
    assert reader.restart_count == 1


@pytest.mark.parametrize("integration", [1, 100, 300, 1000, 10_000_000])
def test_adjustable_exposure_preserves_exact_microseconds(integration: int) -> None:
    assert normalize_integration_us(integration) == integration


@pytest.mark.parametrize("integration", [0, -1, 10_000_001, 300.5, "300", None, True])
def test_invalid_exposure_is_rejected(integration: object) -> None:
    with pytest.raises(ValueError, match="unsupported_integration_us"):
        normalize_integration_us(integration)


@pytest.mark.parametrize("sensor_mode", [-1, 2, "0", None, True])
def test_unknown_sensor_mode_is_rejected(sensor_mode: object) -> None:
    with pytest.raises(ValueError, match="unsupported_sensor_mode"):
        normalize_sensor_mode(sensor_mode)


def test_frame_is_rejected_when_helper_echoes_a_different_mode(tmp_path: Path) -> None:
    reader = BaySpecSdkLiveReader(bridge=_BridgeStub(), app_root=tmp_path)
    reader.last_status = {"sensor_mode": 1, "integration": 300}

    accepted = reader._handle_message(  # noqa: SLF001 - helper attestation test.
        {"type": "spectrum", "counts": [100.0] * 512}
    )

    assert accepted is False
    assert reader.status()["last_error"] == (
        "SDK spectrum rejected after sensor mode mismatch"
    )


def test_x86_helper_maps_mode_zero_to_high_sensitivity() -> None:
    source = (APP_ROOT / "sdk_probe" / "BaySpecSdkStream.cs").read_text(
        encoding="utf-8"
    )

    assert 'IntArg(args, "--sensor-mode", 0)' in source
    assert '? "High Sensitivity"' in source
    assert '"sensor_mode_label"' in source or '\\"sensor_mode_label\\"' in source
    assert "integration < 1 || integration > 10000000" in source
