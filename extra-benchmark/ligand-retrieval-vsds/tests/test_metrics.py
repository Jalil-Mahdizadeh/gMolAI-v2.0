from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from metrics import (  # noqa: E402
    candidate_mask,
    compute_metrics,
    deterministic_anchor_sample,
    fractional_ef_at_fraction,
    tie_averaged_bedroc,
)


class MetricTests(unittest.TestCase):
    def test_realized_fraction_not_fixed_one_percent(self) -> None:
        labels = np.zeros(101, dtype=np.int8)
        labels[:10] = 1
        scores = np.arange(101, 0, -1, dtype=np.float64)
        result = fractional_ef_at_fraction(scores, labels)
        self.assertEqual(result["cutoff_k"], 2)
        self.assertAlmostEqual(result["realized_screened_fraction"], 2 / 101)
        self.assertAlmostEqual(result["ef1"], (2 / 10) / (2 / 101))

    def test_boundary_tie_receives_fractional_credit(self) -> None:
        scores = np.asarray([1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        labels = np.asarray([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        result = fractional_ef_at_fraction(scores, labels)
        self.assertEqual(result["cutoff_k"], 1)
        self.assertEqual(result["boundary_tie_count"], 2)
        self.assertAlmostEqual(result["actives_at_cutoff_fractional"], 0.5)
        self.assertAlmostEqual(result["ef1"], 2.5)

    def test_bedroc_matches_rdkit_without_ties(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float64)
        labels = np.asarray([1, 0, 1, 0, 0, 1], dtype=np.int8)
        ordered = [[float(score), int(label)] for score, label in zip(scores, labels)]
        expected = CalcBEDROC(ordered, 1, 20.0)
        self.assertAlmostEqual(tie_averaged_bedroc(scores, labels), expected, places=14)

    def test_bedroc_is_invariant_to_permutation_inside_ties(self) -> None:
        scores = np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 0.1])
        labels_a = np.asarray([1, 0, 1, 0, 1, 0])
        labels_b = np.asarray([0, 1, 1, 1, 0, 0])
        self.assertAlmostEqual(
            tie_averaged_bedroc(scores, labels_a),
            tie_averaged_bedroc(scores, labels_b),
            places=14,
        )

    def test_sklearn_metrics_are_used_exactly(self) -> None:
        scores = np.asarray([0.7, 0.7, 0.5, 0.2, 0.1])
        labels = np.asarray([1, 0, 1, 0, 0])
        result = compute_metrics(scores, labels)
        self.assertAlmostEqual(result["roc_auc"], roc_auc_score(labels, scores))
        self.assertAlmostEqual(
            result["average_precision"], average_precision_score(labels, scores)
        )

    def test_scaffold_exclusion_removes_both_labels_and_not_empty_scaffolds(self) -> None:
        labels = np.asarray([1, 1, 0, 1, 0, 0])
        scaffolds = np.asarray(["ring-A", "ring-B", "ring-A", "", "", "ring-C"])
        mask = candidate_mask(labels, scaffolds, [0, 3], scaffold_excluded=True)
        self.assertEqual(mask.tolist(), [False, True, False, False, True, True])

    def test_anchors_are_active_removed_and_deterministic(self) -> None:
        active_ids = tuple(f"id-{index}" for index in range(12))
        first = deterministic_anchor_sample(
            active_ids, target_id="P00001", shots=5, draw_id=3, master_seed=20260822
        )
        second = deterministic_anchor_sample(
            tuple(reversed(active_ids)),
            target_id="P00001",
            shots=5,
            draw_id=3,
            master_seed=20260822,
        )
        self.assertEqual(first, second)
        labels = np.ones(12, dtype=np.int8)
        scaffold = np.asarray([""] * 12)
        positions = [active_ids.index(identity) for identity in first[1]]
        mask = candidate_mask(labels, scaffold, positions, scaffold_excluded=False)
        self.assertFalse(mask[positions].any())
        self.assertEqual(int(mask.sum()), 7)

    def test_random_ef_expectation_is_approximately_one(self) -> None:
        generator = np.random.default_rng(7813)
        labels = np.asarray([1] * 10 + [0] * 190, dtype=np.int8)
        values = []
        for _ in range(3000):
            scores = generator.random(labels.size)
            values.append(float(fractional_ef_at_fraction(scores, labels)["ef1"]))
        self.assertTrue(math.isclose(float(np.mean(values)), 1.0, abs_tol=0.12))


if __name__ == "__main__":
    unittest.main()

