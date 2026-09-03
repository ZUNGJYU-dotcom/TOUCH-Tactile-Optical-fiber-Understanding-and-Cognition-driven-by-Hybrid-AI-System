from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_spectrum.px6d_session_dataset import (  # noqa: E402
    SessionDescriptor,
    _load_session_frame_matrix,
    _load_session_recorded_baseline,
    assign_session_folds,
    extract_baseline_relative_features,
    filter_session_descriptors,
    session_has_force_reference,
    split_primary_and_challenge_sessions,
    validate_strict_source_contract,
)


class Px6dSessionDatasetTests(unittest.TestCase):
    def test_force_reference_detection_does_not_treat_blank_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            session_dir = Path(temporary_root)
            descriptor = SessionDescriptor(
                session_dir=session_dir,
                session_id="20260803_test",
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
            )
            (session_dir / "frame_summary.csv").write_text(
                "capture_index,force_fz_n\n1,\n2,\n",
                encoding="utf-8",
            )
            self.assertFalse(session_has_force_reference(descriptor))
            (session_dir / "frame_summary.csv").write_text(
                "capture_index,force_fz_n\n1,\n2,0.25\n",
                encoding="utf-8",
            )
            self.assertTrue(session_has_force_reference(descriptor))

    def test_session_prefix_filter_isolates_collection_batch(self) -> None:
        descriptors = tuple(
            SessionDescriptor(
                session_dir=Path(session_id),
                session_id=session_id,
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
            )
            for session_id in (
                "20260731_P11_old",
                "20260803_P11_current",
            )
        )

        selected = filter_session_descriptors(
            descriptors,
            {"include_session_id_prefixes": ["20260803_"]},
        )

        self.assertEqual(
            [descriptor.session_id for descriptor in selected],
            ["20260803_P11_current"],
        )

    def test_software_build_filter_isolates_same_day_capture_batches(self) -> None:
        descriptors = (
            SessionDescriptor(
                session_dir=Path("older"),
                session_id="20260902_P11_older",
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
                software_version="0.19.16-beta",
                software_build_id="older-build",
            ),
            SessionDescriptor(
                session_dir=Path("current"),
                session_id="20260902_P11_current",
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
                software_version="0.19.19-beta",
                software_build_id="joint-signature-build",
            ),
        )

        selected = filter_session_descriptors(
            descriptors,
            {
                "include_session_id_prefixes": ["20260902_"],
                "include_software_versions": ["0.19.19-beta"],
                "include_software_build_ids": ["joint-signature-build"],
            },
        )

        self.assertEqual(
            [descriptor.session_id for descriptor in selected],
            ["20260902_P11_current"],
        )

    def test_session_loader_sorts_capture_and_point_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            session_dir = Path(temporary_root)
            (session_dir / "frame_summary.csv").write_text(
                "capture_index,elapsed_time_sec,force_fz_n\n"
                "2,0.2,0.4\n"
                "1,0.1,0.0\n",
                encoding="utf-8",
            )
            (session_dir / "spectrum_timeseries.csv").write_text(
                "capture_index,point_index,wavelength_nm,intensity_counts\n"
                "2,2,1541,22\n"
                "1,2,1541,12\n"
                "2,1,1540,21\n"
                "1,1,1540,11\n",
                encoding="utf-8",
            )
            descriptor = SessionDescriptor(
                session_dir=session_dir,
                session_id="20260731_test",
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
            )

            summary, wavelength, intensity = _load_session_frame_matrix(
                descriptor,
                expected_points=2,
            )

            self.assertEqual(summary["capture_index"].tolist(), [1, 2])
            self.assertTrue(np.array_equal(wavelength, [1540.0, 1541.0]))
            self.assertTrue(
                np.array_equal(intensity, [[11.0, 12.0], [21.0, 22.0]])
            )

    def test_recorded_baseline_loader_uses_first_complete_frame_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            session_dir = Path(temporary_root)
            (session_dir / "spectrum_timeseries.csv").write_text(
                "capture_index,point_index,baseline_intensity_counts\n"
                "2,2,902\n"
                "1,2,102\n"
                "2,1,901\n"
                "1,1,101\n",
                encoding="utf-8",
            )
            descriptor = SessionDescriptor(
                session_dir=session_dir,
                session_id="20260803_test",
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
            )

            baseline = _load_session_recorded_baseline(
                descriptor, expected_points=2
            )

            self.assertTrue(np.array_equal(baseline, [101.0, 102.0]))

    def test_recorded_baseline_loader_rejects_zero_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            session_dir = Path(temporary_root)
            (session_dir / "spectrum_timeseries.csv").write_text(
                "capture_index,point_index,baseline_intensity_counts\n"
                "1,1,0\n"
                "1,2,0\n",
                encoding="utf-8",
            )
            descriptor = SessionDescriptor(
                session_dir=session_dir,
                session_id="20260803_test",
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
            )

            self.assertIsNone(
                _load_session_recorded_baseline(descriptor, expected_points=2)
            )

    def test_feature_sets_have_expected_shapes(self) -> None:
        wavelength = np.linspace(1523.0, 1614.0, 512)
        baseline = 1000.0 + 100.0 * np.sin(wavelength)
        intensity = np.vstack((baseline, baseline * 0.95, baseline * 1.05))

        matrix, names, feature_sets = extract_baseline_relative_features(
            intensity,
            baseline,
            wavelength,
            bin_count=64,
        )

        self.assertEqual(matrix.shape, (3, 264))
        self.assertEqual(len(names), 264)
        self.assertEqual(len(feature_sets["baseline_relative_192"]), 192)
        self.assertEqual(len(feature_sets["baseline_relative_264"]), 264)
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertTrue(np.allclose(matrix[0, :128], 0.0, atol=1.0e-12))

    def test_fold_assignment_is_grouped_and_stratified_by_session(self) -> None:
        descriptors = []
        for position in ("P11", "P12", "unlabeled"):
            for index in range(5):
                session_id = f"{position}_{index}"
                descriptors.append(
                    SessionDescriptor(
                        session_dir=Path(session_id),
                        session_id=session_id,
                        trial_id=str(index + 1),
                        position_label=position,
                        qa_status="pass",
                        finding_codes=(),
                    )
                )

        assignments = assign_session_folds(
            descriptors,
            n_splits=5,
            random_seed=42,
        )

        self.assertEqual(len(assignments), 15)
        for position in ("P11", "P12", "unlabeled"):
            self.assertEqual(
                {
                    assignments[f"{position}_{index}"]
                    for index in range(5)
                },
                set(range(5)),
            )

    def test_latest_sessions_are_primary_and_earlier_sessions_are_challenge(
        self,
    ) -> None:
        descriptors = []
        for position, count in (("P11", 10), ("P12", 10), ("P21", 11)):
            for index in range(count):
                descriptors.append(
                    SessionDescriptor(
                        session_dir=Path(f"{position}_{index}"),
                        session_id=f"{position}_{index:02d}",
                        trial_id=str(index),
                        position_label=position,
                        qa_status="pass",
                        finding_codes=(),
                        started_at_epoch_sec=float(index),
                    )
                )
        for index in range(5):
            descriptors.append(
                SessionDescriptor(
                    session_dir=Path(f"P22_{index}"),
                    session_id=f"P22_{index:02d}",
                    trial_id=str(index),
                    position_label="P22",
                    qa_status="pass",
                    finding_codes=(),
                    started_at_epoch_sec=float(index),
                )
            )

        primary, challenge = split_primary_and_challenge_sessions(
            descriptors,
            {
                "mode": "latest_n_by_position",
                "latest_n_by_position": {
                    "P11": 5,
                    "P12": 5,
                    "P21": 5,
                },
                "include_all_other_positions": True,
            },
        )

        self.assertEqual(len(primary), 20)
        self.assertEqual(len(challenge), 16)
        primary_ids = {row.session_id for row in primary}
        challenge_ids = {row.session_id for row in challenge}
        self.assertTrue(primary_ids.isdisjoint(challenge_ids))
        self.assertIn("P11_09", primary_ids)
        self.assertNotIn("P11_04", primary_ids)
        self.assertIn("P11_04", challenge_ids)
        self.assertIn("P21_05", challenge_ids)
        self.assertIn("P21_06", primary_ids)
        self.assertTrue(
            {f"P22_{index:02d}" for index in range(5)} <= primary_ids
        )

    def test_strict_source_contract_rejects_historical_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            capture_root = Path(temporary_root) / "new data"
            session_dir = capture_root / "session"
            session_dir.mkdir(parents=True)
            descriptor = SessionDescriptor(
                session_dir=session_dir,
                session_id="20260730_P11_old",
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
            )
            with self.assertRaisesRegex(ValueError, "strict source contract"):
                validate_strict_source_contract(
                    capture_root,
                    (descriptor,),
                    {
                        "required_capture_root_name": "new data",
                        "required_session_id_prefix": "20260731_",
                        "require_qa_for_every_session": True,
                    },
                )

    def test_strict_source_contract_accepts_current_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            capture_root = Path(temporary_root) / "new data"
            session_dir = capture_root / "session"
            session_dir.mkdir(parents=True)
            descriptor = SessionDescriptor(
                session_dir=session_dir,
                session_id="20260731_P11_current",
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
            )
            validate_strict_source_contract(
                capture_root,
                (descriptor,),
                {
                    "required_capture_root_name": "new data",
                    "required_session_id_prefix": "20260731_",
                    "require_qa_for_every_session": True,
                },
            )

    def test_strict_source_contract_rejects_acquisition_domain_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            capture_root = Path(temporary_root) / "new data"
            session_dir = capture_root / "session"
            session_dir.mkdir(parents=True)
            descriptor = SessionDescriptor(
                session_dir=session_dir,
                session_id="20260831_P11_current",
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
                software_version="0.19.11-beta",
                software_build_id="older-exposure-build",
                optical_source="direct_bayspec_usb20bs_sdk_helper",
                integration_us=5000,
            )

            with self.assertRaisesRegex(ValueError, "expected software version"):
                validate_strict_source_contract(
                    capture_root,
                    (descriptor,),
                    {
                        "required_capture_root_name": "new data",
                        "required_session_id_prefix": "20260831_",
                        "required_software_version": "0.19.12-beta",
                        "required_software_build_id": (
                            "beta-bayspec-high-sensitivity-300us-v19-12-20260831"
                        ),
                        "required_optical_source": (
                            "direct_bayspec_usb20bs_sdk_helper"
                        ),
                        "required_integration_us": 300,
                        "require_qa_for_every_session": True,
                    },
                )

    def test_strict_source_contract_accepts_matching_acquisition_domain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            capture_root = Path(temporary_root) / "new data"
            session_dir = capture_root / "session"
            session_dir.mkdir(parents=True)
            descriptor = SessionDescriptor(
                session_dir=session_dir,
                session_id="20260831_P11_current",
                trial_id="1",
                position_label="P11",
                qa_status="pass",
                finding_codes=(),
                software_version="0.19.12-beta",
                software_build_id=(
                    "beta-bayspec-high-sensitivity-300us-v19-12-20260831"
                ),
                optical_source="direct_bayspec_usb20bs_sdk_helper",
                integration_us=300,
            )

            validate_strict_source_contract(
                capture_root,
                (descriptor,),
                {
                    "required_capture_root_name": "new data",
                    "required_session_id_prefix": "20260831_",
                    "required_software_version": "0.19.12-beta",
                    "required_software_build_id": (
                        "beta-bayspec-high-sensitivity-300us-v19-12-20260831"
                    ),
                    "required_optical_source": (
                        "direct_bayspec_usb20bs_sdk_helper"
                    ),
                    "required_integration_us": 300,
                    "require_qa_for_every_session": True,
                },
            )


if __name__ == "__main__":
    unittest.main()
