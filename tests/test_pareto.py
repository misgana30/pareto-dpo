import pytest
import numpy as np
from pareto_dpo.optimization.pareto import is_pareto_dominant, build_pareto_preference_pairs
from pareto_dpo.optimization.scorer import compute_qed, compute_clogp, compute_sa, compute_mw


class TestParetoDominance:
    def test_dominates(self):
        a = np.array([0.8, 0.5, 0.3], dtype=np.float32)
        b = np.array([0.6, 0.3, 0.2], dtype=np.float32)
        assert is_pareto_dominant(a, b)

    def test_does_not_dominate(self):
        a = np.array([0.6, 0.5, 0.3], dtype=np.float32)
        b = np.array([0.8, 0.3, 0.2], dtype=np.float32)
        assert not is_pareto_dominant(a, b)

    def test_equal_not_dominant(self):
        a = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        assert not is_pareto_dominant(a, b)


class TestScorer:
    def test_qed(self):
        score = compute_qed("c1ccccc1")
        assert score is not None
        assert 0 <= score <= 1

    def test_clogp(self):
        score = compute_clogp("c1ccccc1")
        assert score is not None

    def test_sa(self):
        score = compute_sa("c1ccccc1")
        assert score is not None

    def test_mw(self):
        score = compute_mw("c1ccccc1")
        assert score is not None
        assert score > 0

    def test_invalid_smiles(self):
        assert compute_qed("invalid") is None


class TestBuildPairs:
    def test_empty_input(self):
        pairs = build_pareto_preference_pairs(
            [], ["qed", "clogp"], ["max", "min"]
        )
        assert len(pairs) == 0

    def test_single_molecule(self):
        pairs = build_pareto_preference_pairs(
            [("c1ccccc1", ["c1ccccc1CC(=O)O"])],
            ["qed", "clogp"],
            ["max", "min"],
        )
        assert len(pairs) == 0
