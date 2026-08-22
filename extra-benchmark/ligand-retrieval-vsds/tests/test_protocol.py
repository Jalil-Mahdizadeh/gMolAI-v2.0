from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))

    def test_protocol_is_frozen_and_bounded(self) -> None:
        self.assertEqual(
            self.protocol["protocol_status"],
            "frozen_before_representation_execution",
        )
        self.assertEqual(self.protocol["study"]["dataset_scope"], "VSDS-vd v3 TrueDecoy_gap only")
        self.assertTrue(self.protocol["study"]["no_training_or_finetuning"])
        self.assertTrue(self.protocol["study"]["no_model_or_representation_selection"])

    def test_exact_seven_model_panel(self) -> None:
        expected = [
            "gmolai", "morgan", "molai", "molformer", "smi_ted",
            "molclr_gin", "kermt_v2",
        ]
        self.assertEqual(self.protocol["models"]["primary_order"], expected)
        self.assertEqual(len(expected), 7)

    def test_final_amendments_are_literal(self) -> None:
        ef = self.protocol["metrics"]["ef1"]
        self.assertEqual(ef["k"], "max(1, ceil(0.01*N))")
        self.assertEqual(ef["formula"], "(A_k/A)/(k/N)")
        self.assertIn("fractional", ef["boundary_ties"])
        eligibility = self.protocol["coverage_and_eligibility"]
        self.assertEqual(eligibility["primary_target_minimum_common_actives"], 10)
        self.assertIn("representation export", eligibility["common_support"])

    def test_scaffold_policy_is_prespecified(self) -> None:
        eligibility = self.protocol["coverage_and_eligibility"]
        self.assertEqual(eligibility["scaffold_draw_minimum_remaining_actives"], 5)
        self.assertEqual(eligibility["scaffold_draw_minimum_remaining_inactives"], 1)
        self.assertEqual(eligibility["scaffold_draw_minimum_remaining_inactives"], 1)
        self.assertEqual(eligibility["scaffold_target_minimum_eligible_draws"], 10)
        self.assertIn("active and inactive", self.protocol["retrieval"]["scaffold_exclusion"])
        self.assertIn("never used", self.protocol["chemistry_policy"]["empty_scaffold_policy"])

    def test_source_and_model_hashes_are_sha256(self) -> None:
        hashes = [
            self.protocol["data"]["archive"]["sha256"],
            self.protocol["data"]["gap_definition"]["sha256"],
        ]
        hashes.extend(
            self.protocol["models"][model]["container_sha256"]
            for model in self.protocol["models"]["primary_order"]
        )
        self.assertTrue(all(len(value) == 64 for value in hashes))
        self.assertTrue(all(set(value) <= set("0123456789abcdef") for value in hashes))


if __name__ == "__main__":
    unittest.main()

