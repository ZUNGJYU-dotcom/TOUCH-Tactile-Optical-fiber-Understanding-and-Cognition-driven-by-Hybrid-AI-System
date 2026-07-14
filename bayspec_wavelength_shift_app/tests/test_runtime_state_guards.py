from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from backend import main as backend_main
from backend.main import SenseExportWatcher, _model_display_source_gate
from bridge import BaySpecWavelengthShiftBridge
from desktop_launcher import health_payload_is_expected


class _AliveThread:
    def is_alive(self) -> bool:
        return True


class ModelDisplaySourceGateTests(unittest.TestCase):
    def test_replay_is_allowed_when_live_sources_are_stopped(self) -> None:
        gate = _model_display_source_gate(
            {"source": "static_http_ingest"},
            {"active": False, "freshness": "stopped"},
            {"active": False, "freshness": "stopped"},
        )
        self.assertTrue(gate["model_input_source_allowed"])
        self.assertFalse(gate["source_fresh"])
        self.assertEqual(gate["model_input_source_mode"], "held_replay_or_http")

    def test_stale_sdk_frame_is_blocked(self) -> None:
        gate = _model_display_source_gate(
            {"source": "bayspec_direct_usb20bs_sdk"},
            {"active": False, "freshness": "stopped"},
            {"active": True, "freshness": "stale"},
        )
        self.assertFalse(gate["model_input_source_allowed"])
        self.assertEqual(gate["selected_live_source"], "sdk")

    def test_fresh_sdk_frame_is_allowed(self) -> None:
        gate = _model_display_source_gate(
            {"source": "bayspec_direct_usb20bs_sdk"},
            {"active": False, "freshness": "stopped"},
            {"active": True, "freshness": "live"},
        )
        self.assertTrue(gate["model_input_source_allowed"])
        self.assertTrue(gate["source_fresh"])

    def test_mismatched_buffered_source_is_blocked_during_live_session(self) -> None:
        gate = _model_display_source_gate(
            {"source": "static_http_ingest"},
            {"active": True, "freshness": "live"},
            {"active": False, "freshness": "stopped"},
        )
        self.assertFalse(gate["model_input_source_allowed"])
        self.assertEqual(gate["selected_live_source"], "unmatched_live_source")


class ExportWatcherSessionTests(unittest.TestCase):
    def test_start_clears_previous_session_freshness(self) -> None:
        watcher = SenseExportWatcher()
        watcher.thread = _AliveThread()
        watcher.last_signature = ("old.csv", 1, 2)
        watcher.last_ingest_time = 123.0
        watcher.last_attempt_time = 123.0
        watcher.last_file = "old.csv"
        watcher.last_file_mtime = 123.0
        watcher.last_result = {"ok": True}
        watcher.ingest_count = 99

        status = watcher.start("P22", None, 0.35)

        self.assertEqual(status["freshness"], "waiting_for_export")
        self.assertIsNone(status["last_ingest_time"])
        self.assertIsNone(status["last_file"])
        self.assertEqual(status["ingest_count"], 0)
        self.assertEqual(watcher.last_signature, ("old.csv", 1, 2))

    def test_configuration_change_starts_clean_session(self) -> None:
        watcher = SenseExportWatcher()
        watcher.thread = _AliveThread()
        watcher.active = True
        watcher.channel_id = "P22"
        watcher.export_root = "old-root"
        watcher.last_signature = ("old.csv", 1, 2)
        watcher.last_ingest_time = time.time()

        status = watcher.start("P23", "new-root", 0.35)

        self.assertEqual(status["channel_id"], "P23")
        self.assertEqual(status["freshness"], "waiting_for_export")
        self.assertIsNone(watcher.last_signature)
        self.assertGreater(status["acquisition_session_id"], 0)

    def test_old_export_file_cannot_be_reported_live(self) -> None:
        watcher = SenseExportWatcher()
        watcher.active = True
        watcher.last_ingest_time = time.time()
        watcher.last_file_mtime = time.time() - 120.0

        status = watcher.status()

        self.assertEqual(status["freshness"], "stale")
        self.assertGreater(status["seconds_since_last_file_update"], 100.0)


class AcquisitionSourceMutualExclusionTests(unittest.TestCase):
    def test_export_watch_start_stops_sdk_first(self) -> None:
        watcher_status = {"active": True, "freshness": "waiting_for_export"}
        sdk_status = {"active": False, "freshness": "stopped"}
        with patch.object(backend_main.sdk_live_reader, "stop", return_value=sdk_status) as stop_sdk, patch.object(
            backend_main.export_watcher, "start", return_value=watcher_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.export_watch_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.35,
            )

        stop_sdk.assert_called_once_with()
        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertTrue(result["acquisition_session_reset"]["baseline_invalidated"])
        self.assertEqual(result["sdk_live"], sdk_status)

    def test_live_export_source_stops_sdk_first(self) -> None:
        watcher_status = {"active": True, "freshness": "waiting_for_export"}
        sdk_status = {"active": False, "freshness": "stopped"}
        with patch.object(backend_main.sdk_live_reader, "stop", return_value=sdk_status) as stop_sdk, patch.object(
            backend_main.export_watcher, "start", return_value=watcher_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.live_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.35,
                control_sense=False,
                source="export_watch",
            )

        stop_sdk.assert_called_once_with()
        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertEqual(result["sdk_live"], sdk_status)

    def test_sdk_start_invalidates_previous_session_baseline(self) -> None:
        sdk_status = {"active": True, "freshness": "waiting"}
        with patch.object(backend_main.export_watcher, "stop"), patch.object(
            backend_main.sdk_live_reader, "start", return_value=sdk_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.sdk_start(channel_id="P22", interval_ms=100, integration=40000)

        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertTrue(result["acquisition_session_reset"]["baseline_invalidated"])

    def test_direct_live_start_invalidates_previous_session_baseline(self) -> None:
        sdk_status = {"active": True, "freshness": "waiting"}
        with patch.object(backend_main.export_watcher, "stop"), patch.object(
            backend_main.sdk_live_reader, "start", return_value=sdk_status
        ), patch.object(backend_main.bridge, "reset", return_value={"ok": True}) as reset_bridge:
            result = backend_main.live_start(
                channel_id="P22",
                export_root=None,
                interval_sec=0.1,
                control_sense=False,
                source="direct_sdk",
            )

        reset_bridge.assert_called_once_with(keep_baseline=False)
        self.assertTrue(result["acquisition_session_reset"]["baseline_invalidated"])


class BridgeResetTests(unittest.TestCase):
    def test_keep_baseline_reset_clears_temporal_tracking(self) -> None:
        test_bridge = BaySpecWavelengthShiftBridge(max_records_per_channel=20)
        test_bridge.baseline_wavelength_by_channel["P22"] = 1546.89
        test_bridge.previous_tracked_wavelength_by_channel["P22"] = 1547.10

        result = test_bridge.reset(keep_baseline=True)

        self.assertTrue(result["ok"])
        self.assertEqual(test_bridge.baseline_wavelength_by_channel["P22"], 1546.89)
        self.assertNotIn("P22", test_bridge.previous_tracked_wavelength_by_channel)


class DesktopLauncherIdentityTests(unittest.TestCase):
    def test_expected_backend_identity_is_accepted(self) -> None:
        self.assertTrue(
            health_payload_is_expected(
                {
                    "ok": True,
                    "app": "TOUCH System Trained Static Spectrum Twin",
                    "mode": "standalone_bayspec_trained_static_spectrum_twin",
                    "trained_static_model_primary": True,
                }
            )
        )

    def test_other_touch_backend_is_rejected(self) -> None:
        self.assertFalse(
            health_payload_is_expected(
                {
                    "ok": True,
                    "app": "TOUCH System Optical Intensity Twin",
                    "mode": "standalone_bayspec_optical_intensity",
                    "trained_static_model_primary": False,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
