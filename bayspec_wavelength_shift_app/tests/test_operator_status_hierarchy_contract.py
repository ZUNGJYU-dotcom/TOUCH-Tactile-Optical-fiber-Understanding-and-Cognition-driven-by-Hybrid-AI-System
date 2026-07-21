from pathlib import Path
import unittest


APP_ROOT = Path(__file__).resolve().parents[1]


class OperatorStatusHierarchyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (APP_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        cls.css = (APP_ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
        cls.js = (APP_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

    def test_operator_qa_uses_compact_tokens_only(self) -> None:
        qa_start = self.js.index("function operatorQaLabel")
        qa_end = self.js.index("function operatorFreshnessLabel", qa_start)
        qa_function = self.js[qa_start:qa_end]

        for token in ("OK", "STALE", "BASELINE", "CHECK", "REVIEW"):
            self.assertIn(f'return "{token}"', qa_function)
        self.assertNotIn('return status.replace(/_/g, " ")', qa_function)
        self.assertNotIn("Model confidence warning", qa_function)
        self.assertNotIn("OK · manual λ", qa_function)

    def test_full_qa_detail_is_routed_to_diagnostics(self) -> None:
        self.assertIn("Open Diagnostics for details.", self.js)
        self.assertIn('operatorDiagnosticsButton?.classList.toggle("attention-required"', self.js)
        self.assertIn('operatorAlert.classList.remove("visible")', self.js)
        self.assertIn("details are available in Diagnostics", self.html)

    def test_compact_qa_slot_releases_topbar_width(self) -> None:
        self.assertIn("flex: 0 1 112px", self.css)
        self.assertIn("flex-basis: 96px", self.css)

    def test_operator_coupling_status_uses_a_compact_value(self) -> None:
        self.assertIn('? "coupled"', self.js)
        self.assertNotIn('? "global mixed spectral fingerprint"', self.js)


if __name__ == "__main__":
    unittest.main()
