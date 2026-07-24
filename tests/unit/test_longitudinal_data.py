"""Unit tests for the validated longitudinal-data containers."""

import numpy as np
import pandas as pd
import pytest

from longmi import (
    AnalysisEstimate,
    CallableAnalysis,
    CompletedDatasetCollection,
    LongitudinalData,
)


def make_frame():
    return pd.DataFrame(
        {
            "pid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "wave": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y": [4.0, 2.0, np.nan, 7.0, np.nan, np.nan, 1.0, 0.0, 3.0],
            "treat": [0, 0, 0, 1, 1, 1, 1, 1, 1],
            "age": [30.0, 30.0, 30.0, 41.0, 41.0, 41.0, 25.0, 25.0, 25.0],
        }
    )


def make_data(**overrides):
    kwargs = dict(
        id_col="pid",
        time_col="wave",
        outcome_col="y",
        predictor_cols=("treat", "age"),
        outcome_type="count",
    )
    kwargs.update(overrides)
    return LongitudinalData(make_frame(), **kwargs)


class TestValidation:
    def test_valid_construction(self):
        data = make_data()
        assert data.n_rows == 9
        assert data.n_participants == 3
        assert data.n_missing == 3
        assert not data.is_complete
        assert data.observed_times() == (1, 2, 3)

    def test_missing_mask_marks_exactly_missing_outcomes(self):
        data = make_data()
        frame = data.frame
        mask = data.missing_mask
        assert int(mask.sum()) == 3
        missing_pairs = set(
            zip(frame.loc[mask, "pid"], frame.loc[mask, "wave"])
        )
        assert missing_pairs == {(1, 3), (2, 2), (2, 3)}

    def test_duplicate_id_time_rejected(self):
        frame = make_frame()
        frame.loc[1, "wave"] = 1  # participant 1 now has two wave-1 rows
        with pytest.raises(ValueError, match="duplicate"):
            LongitudinalData(
                frame,
                id_col="pid",
                time_col="wave",
                outcome_col="y",
                predictor_cols=("treat", "age"),
            )

    def test_missing_predictor_rejected(self):
        frame = make_frame()
        frame.loc[4, "age"] = np.nan
        with pytest.raises(ValueError, match="fully observed predictors"):
            LongitudinalData(
                frame,
                id_col="pid",
                time_col="wave",
                outcome_col="y",
                predictor_cols=("treat", "age"),
            )

    def test_missing_time_rejected(self):
        frame = make_frame()
        frame.loc[0, "wave"] = np.nan
        with pytest.raises(ValueError, match="time column"):
            LongitudinalData(
                frame, id_col="pid", time_col="wave", outcome_col="y"
            )

    def test_undeclared_design_time_rejected(self):
        with pytest.raises(ValueError, match="design times"):
            make_data(times=(1, 2))

    def test_incomplete_wave_grid_is_rejected(self):
        frame = make_frame().drop(index=4)  # participant 2 loses its wave-2 row
        with pytest.raises(ValueError, match="grid is incomplete"):
            LongitudinalData(
                frame,
                id_col="pid",
                time_col="wave",
                outcome_col="y",
                times=(1, 2, 3),
            )

    def test_incomplete_wave_grid_opt_out(self):
        frame = make_frame().drop(index=4)
        data = LongitudinalData(
            frame,
            id_col="pid",
            time_col="wave",
            outcome_col="y",
            times=(1, 2, 3),
            require_complete_grid=False,
        )
        assert data.n_rows == 8

    def test_nonfinite_outcome_is_rejected(self):
        frame = make_frame()
        frame.loc[0, "y"] = np.inf
        for outcome_type in ("count", "continuous"):
            with pytest.raises(ValueError, match="non-finite"):
                LongitudinalData(
                    frame,
                    id_col="pid",
                    time_col="wave",
                    outcome_col="y",
                    outcome_type=outcome_type,
                )

    def test_observed_times_preserves_meaningful_order(self):
        frame = make_frame()
        frame["wave"] = frame["wave"].map({1: 0, 2: 3, 3: 12})
        data = LongitudinalData(
            frame, id_col="pid", time_col="wave", outcome_col="y"
        )
        assert data.observed_times() == (0, 3, 12)  # numeric, not lexicographic
        declared = LongitudinalData(
            frame,
            id_col="pid",
            time_col="wave",
            outcome_col="y",
            times=(0, 3, 12),
        )
        assert declared.observed_times() == (0, 3, 12)

    def test_negative_count_rejected(self):
        frame = make_frame()
        frame.loc[0, "y"] = -1.0
        with pytest.raises(ValueError, match="negative"):
            LongitudinalData(
                frame,
                id_col="pid",
                time_col="wave",
                outcome_col="y",
                outcome_type="count",
            )

    def test_noninteger_count_rejected(self):
        frame = make_frame()
        frame.loc[0, "y"] = 2.5
        with pytest.raises(ValueError, match="non-integer"):
            LongitudinalData(
                frame,
                id_col="pid",
                time_col="wave",
                outcome_col="y",
                outcome_type="count",
            )

    def test_noninteger_continuous_accepted(self):
        frame = make_frame()
        frame.loc[0, "y"] = 2.5
        data = LongitudinalData(
            frame, id_col="pid", time_col="wave", outcome_col="y"
        )
        assert data.n_missing == 3

    def test_rows_sorted_deterministically(self):
        frame = make_frame().sample(frac=1.0, random_state=7)
        data = LongitudinalData(
            frame,
            id_col="pid",
            time_col="wave",
            outcome_col="y",
            predictor_cols=("treat", "age"),
        )
        ordered = data.frame
        assert list(zip(ordered["pid"], ordered["wave"])) == sorted(
            zip(make_frame()["pid"], make_frame()["wave"])
        )


class TestCompletion:
    def test_completed_with_fills_every_missing_value(self):
        data = make_data()
        completed = data.completed_with([5.0, 3.0, 0.0])
        assert completed["y"].notna().all()
        # observed values untouched
        observed = ~data.missing_mask
        np.testing.assert_array_equal(
            completed.loc[observed, "y"].to_numpy(),
            data.frame.loc[observed, "y"].to_numpy(),
        )

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="expected 3 imputed values"):
            make_data().completed_with([1.0, 2.0])

    def test_nonfinite_imputation_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            make_data().completed_with([1.0, np.nan, 2.0])

    def test_count_constraints_enforced_on_imputations(self):
        with pytest.raises(ValueError, match="negative"):
            make_data().completed_with([-1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="non-integer"):
            make_data().completed_with([1.5, 2.0, 3.0])
        # continuous outcome accepts the same values
        cont = make_data(outcome_type="continuous")
        cont.completed_with([1.5, 2.0, 3.0])

    def test_check_completed_detects_overwritten_observed_value(self):
        data = make_data()
        completed = data.completed_with([5.0, 3.0, 0.0])
        completed.loc[0, "y"] = 99.0  # tamper with an observed cell
        with pytest.raises(ValueError, match="overwrote observed"):
            data.check_completed(completed)

    def test_check_completed_detects_remaining_missing(self):
        data = make_data()
        completed = data.completed_with([5.0, 3.0, 0.0])
        completed.loc[2, "y"] = np.nan
        with pytest.raises(ValueError, match="missing outcomes"):
            data.check_completed(completed)

    def test_check_completed_detects_altered_predictor(self):
        data = make_data()
        completed = data.completed_with([5.0, 3.0, 0.0])
        completed.loc[0, "treat"] = 1
        with pytest.raises(ValueError, match="altered column"):
            data.check_completed(completed)


class TestCompletedDatasetCollection:
    def test_collection_validates_and_iterates(self):
        data = make_data()
        frames = [
            data.completed_with([5.0, 3.0, 0.0]),
            data.completed_with([4.0, 2.0, 1.0]),
        ]
        collection = CompletedDatasetCollection(data, frames)
        assert collection.m == len(collection) == 2
        for frame in collection:
            assert frame["y"].notna().all()

    def test_collection_rejects_tampered_frame(self):
        data = make_data()
        good = data.completed_with([5.0, 3.0, 0.0])
        bad = good.copy()
        bad.loc[0, "y"] = 42.0
        with pytest.raises(ValueError, match="completed dataset 2"):
            CompletedDatasetCollection(data, [good, bad])

    def test_analysis_model_fit_is_called(self):
        data = make_data()
        frames = [
            data.completed_with([5.0, 3.0, 0.0]),
            data.completed_with([4.0, 2.0, 1.0]),
        ]
        collection = CompletedDatasetCollection(data, frames)

        class MeanModel:
            calls = 0

            def fit(self, frame):
                MeanModel.calls += 1
                y = frame["y"].to_numpy()
                return AnalysisEstimate(
                    names=("mean",),
                    estimates=[y.mean()],
                    covariance=[[y.var(ddof=1) / len(y)]],
                )

        estimates = collection.analyze(MeanModel())
        assert MeanModel.calls == 2
        expected = [f["y"].mean() for f in frames]
        assert [e.estimates[0] for e in estimates] == pytest.approx(expected)

    def test_analyze_rejects_wrong_return_type(self):
        data = make_data()
        collection = CompletedDatasetCollection(
            data, [data.completed_with([5.0, 3.0, 0.0])]
        )
        with pytest.raises(RuntimeError, match="completed dataset 1") as info:
            collection.analyze(CallableAnalysis(lambda f: f["y"].mean()))
        assert isinstance(info.value.__cause__, TypeError)

    def test_declaration_is_carried(self):
        from longmi import ValidityDeclaration

        data = make_data()
        decl = ValidityDeclaration(missingness_assumption="MAR")
        collection = CompletedDatasetCollection(
            data, [data.completed_with([5.0, 3.0, 0.0])], declaration=decl
        )
        assert collection.declaration is decl
