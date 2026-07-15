from __future__ import annotations

import unittest

import numpy as np

from src.hybrid_spectrum.dynamic_twin_mapping import (
    ARRAY_CHANNEL_COORDS,
    ARRAY_DISPLAY_ROWS,
    RESPONSE_DEFORMATION_PROXY,
    dynamic_prediction_to_twin_proxy,
)


def prediction(position: str, response: str, *, contact: str = "contact") -> dict:
    return {
        "ready": True,
        "contact": {"label": contact},
        "position": {"label": position} if contact == "contact" else None,
        "response_level": {"label": response} if contact == "contact" else None,
        "operational_state": "active_contact" if contact == "contact" else "no_contact",
    }


class DynamicTwinMappingTests(unittest.TestCase):
    def test_display_layout_matches_physical_array_contract(self) -> None:
        self.assertEqual(
            ARRAY_DISPLAY_ROWS,
            (
                ("P11", "P21", "P31"),
                ("P12", "P22", "P32"),
                ("P13", "P23", "P33"),
            ),
        )

    def test_every_position_places_grid_max_and_centroid_at_that_position(self) -> None:
        for position_id, coordinate in ARRAY_CHANNEL_COORDS.items():
            with self.subTest(position_id=position_id):
                proxy = dynamic_prediction_to_twin_proxy(
                    prediction(position_id, "normal")
                )
                grid = np.asarray(proxy["surface_grid"])
                row, column = np.unravel_index(np.argmax(grid), grid.shape)
                self.assertEqual(ARRAY_DISPLAY_ROWS[row][column], position_id)
                self.assertEqual(
                    (
                        proxy["surface_metrics"]["surface_centroid_x"],
                        proxy["surface_metrics"]["surface_centroid_y"],
                    ),
                    coordinate,
                )

    def test_light_normal_hard_deformation_is_strictly_monotonic(self) -> None:
        levels = ("light", "normal", "hard")
        peaks = [
            dynamic_prediction_to_twin_proxy(prediction("P22", level))[
                "deformation_proxy"
            ]
            for level in levels
        ]
        self.assertEqual(peaks, [RESPONSE_DEFORMATION_PROXY[level] for level in levels])
        self.assertLess(peaks[0], peaks[1])
        self.assertLess(peaks[1], peaks[2])

    def test_no_contact_and_release_suppress_deformation(self) -> None:
        for contact in ("no_contact", "released"):
            with self.subTest(contact=contact):
                proxy = dynamic_prediction_to_twin_proxy(
                    prediction("P22", "hard", contact=contact)
                )
                self.assertFalse(proxy["active"])
                self.assertEqual(proxy["deformation_proxy"], 0.0)
                self.assertEqual(np.max(proxy["surface_grid"]), 0.0)


if __name__ == "__main__":
    unittest.main()
