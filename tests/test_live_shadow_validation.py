from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from src.hybrid_spectrum.live_shadow_validation import (
    flatten_shadow_frame,
    summarize_shadow_records,
    write_shadow_validation_artifacts,
)


def prediction(position: str, force: str, agreement: float = 1.0) -> dict:
    return {
        "model_bundle_sha256": "bundle",
        "contact": {"label": "contact", "confidence": 0.9},
        "position": {
            "label": position,
            "confidence": agreement,
            "margin": agreement - 0.2,
            "confidence_source": "ensemble_vote_fraction_not_calibrated",
            "ensemble_diagnostics": {
                "agreement_fraction": agreement,
                "unanimous": agreement == 1.0,
                "member_predictions": {"a": position, "b": position, "c": "P22"},
            },
        },
        "force_level": {"label": force, "confidence": 0.8, "margin": 0.5},
        "force_model_scope": "global_manual_fallback",
        "uncertainty": {"review_needed": False},
    }


class LiveShadowValidationTests(unittest.TestCase):
    def test_flattens_same_frame_and_preserves_shadow_boundary(self) -> None:
        frame = {
            "frame_id": 12,
            "source_fresh": True,
            "latest": {"frame_id": 12, "source": "sdk", "timestamp": 4.2},
            "trained_static_spectral_model": {
                "model_bundle_sha256": "primary-hash",
                "shadow_candidate": {"model_bundle_sha256": "shadow-hash"},
            },
            "trained_static_spectral_frame": {
                "status": "ready",
                "input": {"baseline_ready": True, "baseline_spectrum_sample_count": 30},
                "prediction": prediction("P21", "normal"),
                "shadow_candidate": {
                    "ok": True,
                    "status": "shadow_ready",
                    "prediction": prediction("P21", "normal", 2.0 / 3.0),
                    "runtime_role": "shadow_only_not_driving_digital_twin",
                    "drives_operator_ui": False,
                    "drives_digital_twin": False,
                    "temporal_stabilization": {
                        "status": "stable_contact",
                        "ready": True,
                        "contact_label": "contact",
                        "position_label": "P21",
                        "force_label": "normal",
                        "position_support": 1.0,
                        "force_support": 0.8,
                        "contact_votes": 5,
                        "history_unique_frames": 5,
                        "duplicate_frame_ignored": False,
                        "semantics": "unique_frame_temporal_vote_diagnostic_only",
                    },
                },
            },
        }

        row = flatten_shadow_frame(
            frame,
            captured_at="now",
            expected_position="P21",
            expected_force="normal",
        )

        self.assertEqual(row["frame_id"], 12)
        self.assertTrue(row["both_models_ready"])
        self.assertTrue(row["position_models_agree"])
        self.assertTrue(row["shadow_position_correct"])
        self.assertAlmostEqual(row["shadow_position_agreement"], 2.0 / 3.0)
        self.assertFalse(row["shadow_drives_operator_ui"])
        self.assertFalse(row["shadow_drives_digital_twin"])
        self.assertEqual(row["shadow_temporal_status"], "stable_contact")
        self.assertEqual(row["shadow_temporal_position"], "P21")
        self.assertTrue(row["shadow_temporal_position_correct"])

    def test_blocked_frame_is_logged_without_fabricated_predictions(self) -> None:
        row = flatten_shadow_frame(
            {
                "frame_id": 2,
                "latest": {"frame_id": 2, "source": "sdk"},
                "trained_static_spectral_frame": {
                    "status": "baseline_required",
                    "reason": "current_no_contact_full_spectrum_baseline_required",
                    "input": {"baseline_ready": False},
                },
            },
            captured_at="now",
        )

        self.assertFalse(row["both_models_ready"])
        self.assertIsNone(row["primary_position"])
        self.assertIsNone(row["shadow_force"])
        self.assertEqual(row["primary_model_status"], "baseline_required")

    def test_summary_and_artifacts_never_promote_shadow(self) -> None:
        rows = [
            {
                "frame_id": 1,
                "data_source": "sdk",
                "baseline_ready": True,
                "both_models_ready": True,
                "primary_model_status": "ready",
                "shadow_status": "shadow_ready",
                "position_models_agree": True,
                "force_models_agree": False,
                "shadow_position_agreement": 1.0,
                "shadow_position_unanimous": True,
                "shadow_drives_operator_ui": False,
                "shadow_drives_digital_twin": False,
                "shadow_temporal_status": "stable_contact",
                "shadow_temporal_contact_correct": True,
                "shadow_temporal_position_correct": True,
                "shadow_temporal_force_correct": True,
                "primary_contact_correct": True,
                "shadow_contact_correct": True,
                "primary_position_correct": True,
                "shadow_position_correct": True,
                "primary_force_correct": False,
                "shadow_force_correct": True,
            }
        ]
        summary = summarize_shadow_records(rows)
        self.assertEqual(summary["deployment_decision"], "shadow_only_not_promoted")
        self.assertFalse(summary["shadow_ever_drove_operator_ui"])

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            written = write_shadow_validation_artifacts(output_dir, rows)
            self.assertEqual(written["record_count"], 1)
            self.assertTrue((output_dir / "live_shadow_predictions.csv").exists())
            self.assertTrue((output_dir / "live_shadow_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
