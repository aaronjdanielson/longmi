"""PR 1 contracts: binary outcomes and targeted delta rules (deterministic)."""

import numpy as np
import pandas as pd
import pytest

from longmi import DeltaAdjustment, DeltaScenario, LongitudinalData
from longmi.impute import JointGaussianImputer, NegativeBinomialImputer
from longmi.scenarios import realized_deltas


def frame(y=(1.0, 0.0, np.nan, 0.0, 1.0, np.nan)):
    return pd.DataFrame({
        "pid": [1, 1, 1, 2, 2, 2], "wave": [1, 2, 3, 1, 2, 3],
        "y": y, "exposure": [1.0] * 3 + [0.0] * 3,
    })


def make(y=(1.0, 0.0, np.nan, 0.0, 1.0, np.nan), **kw):
    return LongitudinalData(frame(y), id_col="pid", time_col="wave",
                            outcome_col="y", predictor_cols=("exposure",),
                            outcome_type="binary", times=(1, 2, 3), **kw)


class TestBinaryContract:
    def test_valid_construction(self):
        d = make()
        assert d.n_missing == 2 and d.outcome_type == "binary"

    def test_nonbinary_values_rejected(self):
        for bad in (2.0, -1.0, 0.5):
            with pytest.raises(ValueError, match="binary outcome must be 0 or 1"):
                make(y=(bad, 0.0, np.nan, 0.0, 1.0, np.nan))

    def test_boolean_outcomes_normalized(self):
        f = frame()
        f["y"] = pd.array([True, False, None, False, True, None],
                          dtype="boolean")
        d = LongitudinalData(f, id_col="pid", time_col="wave", outcome_col="y",
                             predictor_cols=("exposure",),
                             outcome_type="binary", times=(1, 2, 3))
        assert set(d.frame["y"].dropna()) <= {0.0, 1.0}

    def test_completion_enforces_binary_support(self):
        d = make()
        done = d.completed_with([1.0, 0.0])
        assert set(done["y"]) <= {0.0, 1.0}
        with pytest.raises(ValueError, match="binary"):
            d.completed_with([2.0, 0.0])
        with pytest.raises(ValueError, match="binary"):
            d.completed_with([0.5, 0.0])

    def test_existing_backends_refuse_binary(self):
        d = make()
        with pytest.raises(ValueError, match="continuous outcomes only"):
            JointGaussianImputer().fit(d)
        with pytest.raises(ValueError, match="count outcomes only"):
            NegativeBinomialImputer().fit(d)


class TestTargetedDeltas:
    def test_scalar_rule_applies_to_all_missing(self):
        np.testing.assert_array_equal(
            realized_deltas(DeltaAdjustment(0.5), make()), [0.5, 0.5])

    def test_where_and_times_target_rows(self):
        rule = DeltaAdjustment(np.log(0.8), where={"exposure": 1}, times=(3,))
        # missing rows in stored order: (pid1,w3,exposure=1), (pid2,w3,exposure=0)
        np.testing.assert_allclose(
            realized_deltas(rule, make()), [np.log(0.8), 0.0])

    def test_scenario_combines_disjoint_rules(self):
        sc = DeltaScenario(adjustments=(
            DeltaAdjustment(0.2, where={"exposure": 1}),
            DeltaAdjustment(-0.3, where={"exposure": 0}),
        ), label="group-specific")
        np.testing.assert_allclose(realized_deltas(sc, make()), [0.2, -0.3])

    def test_overlapping_rules_rejected(self):
        sc = DeltaScenario(adjustments=(
            DeltaAdjustment(0.2, times=(3,)),
            DeltaAdjustment(0.1, where={"exposure": 1}),
        ))
        with pytest.raises(ValueError, match="overlapping delta rules"):
            realized_deltas(sc, make())

    def test_bad_references_rejected(self):
        with pytest.raises(ValueError, match="not a\\s+declared predictor"):
            realized_deltas(DeltaAdjustment(0.1, where={"nope": 1}), make())
        with pytest.raises(ValueError, match="unknown times"):
            realized_deltas(DeltaAdjustment(0.1, times=(9,)), make())
        with pytest.raises(ValueError, match="matches no rows"):
            realized_deltas(DeltaAdjustment(0.1, where={"exposure": 7}), make())

    def test_mixed_scales_and_empty_scenarios_rejected(self):
        with pytest.raises(ValueError, match="share a scale"):
            DeltaScenario(adjustments=(
                DeltaAdjustment(0.1, scale="outcome"),
                DeltaAdjustment(0.1, scale="linear_predictor")))
        with pytest.raises(ValueError, match="at least one"):
            DeltaScenario(adjustments=())

    def test_backends_refuse_targeted_rules_until_wired(self):
        rule = DeltaAdjustment(0.1, times=(3,))
        with pytest.raises(NotImplementedError, match="targeted delta"):
            NegativeBinomialImputer(delta=rule)
