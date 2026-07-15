from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.hybrid_spectrum.sense_fast_dat import (
    detect_fast_dat_layout,
    read_sense_fast_dat,
)


def _spectra(frame_count: int = 24) -> np.ndarray:
    x = np.arange(512, dtype=float)
    rows = []
    for frame_index in range(frame_count):
        center = 78.0 + 0.03 * frame_index
        spectrum = 2500.0 + 42000.0 * np.exp(
            -0.5 * ((x - center) / 4.0) ** 2
        )
        spectrum += 27000.0 * np.exp(-0.5 * ((x - 205.0) / 5.0) ** 2)
        rows.append(np.rint(spectrum).astype(">u2"))
    return np.stack(rows)


class SenseFastDatLayoutTests(unittest.TestCase):
    def test_detects_513_word_records_and_discards_auxiliary_word(self) -> None:
        spectra = _spectra()
        auxiliary = (np.arange(len(spectra), dtype=np.uint16) * 4096)[:, None]
        records = np.concatenate([spectra, auxiliary.astype(">u2")], axis=1)
        words = np.concatenate(
            [records.reshape(-1), np.asarray([11, 22, 33], dtype=">u2")]
        )

        layout = detect_fast_dat_layout(words)

        self.assertEqual(layout.record_words, 513)
        self.assertEqual(layout.frame_count, len(spectra))
        self.assertEqual(layout.trailing_words, 3)
        self.assertGreater(layout.median_adjacent_correlation, 0.999)

    def test_detects_legacy_512_word_records(self) -> None:
        spectra = _spectra()
        layout = detect_fast_dat_layout(spectra.reshape(-1))
        self.assertEqual(layout.record_words, 512)
        self.assertEqual(layout.prefix_words, 0)

    def test_detects_legacy_256_word_prefix(self) -> None:
        spectra = _spectra()
        prefix = np.arange(256, dtype=">u2")
        words = np.concatenate([prefix, spectra.reshape(-1)])
        layout = detect_fast_dat_layout(words)
        self.assertEqual(layout.record_words, 512)
        self.assertEqual(layout.prefix_words, 256)

    def test_reader_returns_only_complete_immutable_frames(self) -> None:
        spectra = _spectra(10)
        auxiliary = np.arange(10, dtype=">u2")[:, None]
        records = np.concatenate([spectra, auxiliary], axis=1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sequence.dat"
            np.concatenate(
                [records.reshape(-1), np.asarray([99], dtype=">u2")]
            ).astype(">u2").tofile(path)

            sequence = read_sense_fast_dat(path)

        self.assertEqual(sequence.spectra.shape, (10, 512))
        self.assertEqual(sequence.auxiliary_words.shape, (10, 1))
        self.assertTrue(np.array_equal(sequence.spectra, spectra.astype(np.uint16)))
        self.assertFalse(sequence.spectra.flags.writeable)


if __name__ == "__main__":
    unittest.main()
