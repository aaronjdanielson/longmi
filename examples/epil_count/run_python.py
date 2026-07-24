"""Python side of the epil worked example.

Fits the substantive Poisson GEE

    log E(y_ij) = b0 + bT*treat + bP*period + bTP*treat:period
                  + bB*log(1 + base) + bA*age

with exchangeable working correlation clustered by subject, on

- the complete upstream data (the benchmark),
- the available cases under the shared MAR mask
  (validation/masks/epil_mar_seed_20260723.csv), and
- M = 20 completed datasets from longmi's negative-binomial GLMM imputer
  (categorical wave, treat-by-wave interactions so the analysis's
  treat:period term is nested — A8), pooled with Rubin's rules,

then runs a delta-adjusted MNAR sensitivity analysis (the same fitted
imputation model, means of the imputed counts shifted by exp(delta)).

Because the source data are complete, the complete-data GEE is the truth
the missing-data methods are trying to recover: available-case shows the
cost of ignoring the dropout mechanism; MI should land nearer the
benchmark with honestly wider uncertainty.

Writes machine-readable results to results_python.csv with
language-neutral term names, for comparison against the R reference
(run_reference.R -> results_r.csv, complete/available-case analyses) by
compare_results.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from missingness import apply_mask, mask_path
from prepare_data import load_epil, normalized_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from longmi import LongitudinalData  # noqa: E402
from longmi.analysis import StatsmodelsGEE  # noqa: E402
from longmi.impute import NegativeBinomialImputer  # noqa: E402
from longmi.pooling import pool_rubin  # noqa: E402
from longmi.scenarios import DeltaAdjustment  # noqa: E402

HERE = Path(__file__).resolve().parent

FORMULA = "y ~ treat * period + lbase1 + age"
M_IMPUTATIONS = 20
MI_SEED = 20260723
DELTA_GRID = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4)  # exp(delta) multipliers
IPW_BOOT = 300

# statsmodels term name -> language-neutral name (order fixed by the formula)
TERMS = {
    "Intercept": "intercept",
    "treat": "treat",
    "period": "period",
    "treat:period": "treat_period",
    "lbase1": "lbase1",
    "age": "age",
}


def fit_gee(frame: pd.DataFrame, analysis: str) -> pd.DataFrame:
    model = smf.gee(
        FORMULA,
        groups="subject",
        data=frame,
        family=sm.families.Poisson(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    fit = model.fit()  # robust (sandwich) covariance is the default
    if list(fit.params.index) != list(TERMS):
        raise RuntimeError(f"unexpected term ordering: {list(fit.params.index)}")
    return pd.DataFrame(
        {
            "analysis": analysis,
            "term": list(TERMS.values()),
            "estimate": fit.params.to_numpy(),
            "robust_se": fit.bse.to_numpy(),
            "n_rows": len(frame),
            "n_subjects": frame["subject"].nunique(),
        }
    )


def fit_mi(
    observed: pd.DataFrame,
    analysis: str,
    delta: float | None,
    cache_key: str = "mar",
) -> pd.DataFrame:
    """NB imputation -> GEE per completed dataset -> Rubin pooling."""
    data = LongitudinalData(
        observed,
        id_col="subject",
        time_col="period",
        outcome_col="y",
        predictor_cols=("treat", "lbase1", "age"),
        outcome_type="count",
        times=(1, 2, 3, 4),
    )
    imputer = NegativeBinomialImputer(time_interactions=("treat",))
    fit = fit_mi.cache.get(cache_key)
    if fit is None:
        fit = imputer.fit(data)
        fit_mi.cache[cache_key] = fit
        diag = fit.diagnostics
        print(
            f"NB imputation model [{cache_key}]: optimizer ok "
            f"(grad {diag.gradient_norm:.2e}), "
            f"min Hessian eigenvalue {min(diag.hessian_eigenvalues):.3g}"
        )
    scenario = None if delta is None else DeltaAdjustment(
        delta=delta, label=f"means x {np.exp(delta):.2f}"
    )
    collection = fit.impute(M_IMPUTATIONS, random_state=MI_SEED, delta=scenario)
    adapter = StatsmodelsGEE(
        FORMULA, groups="subject", family="poisson", cov_struct="exchangeable"
    )
    pooled = pool_rubin(collection.analyze(adapter), validity=collection.declaration)
    name_map = dict(zip(pooled.names, TERMS.values()))
    if set(name_map) != set(TERMS):
        raise RuntimeError(f"unexpected MI term ordering: {pooled.names}")
    return pd.DataFrame(
        {
            "analysis": analysis,
            "term": [TERMS[n] for n in pooled.names],
            "estimate": pooled.qbar,
            "robust_se": pooled.se,
            "n_rows": len(observed),
            "n_subjects": observed["subject"].nunique(),
        }
    )


fit_mi.cache = {}


def _stabilized_weights(df: pd.DataFrame) -> pd.Series:
    """Sequential-MAR stabilized IPW under the monotone mask.

    Retention hazard among those still in at period j:
    logit P(R_j = 1) = wave effects + treat + lbase1 + age + log(1 + y_prev).
    Numerator: wave effects only.
    """
    at_risk = (df["period"] > 1) & (df["r_prev"] == 1)
    num = smf.glm(
        "r ~ C(period)", data=df[at_risk], family=sm.families.Binomial()
    ).fit()
    den = smf.glm(
        "r ~ C(period) + treat + lbase1 + age + np.log1p(y_prev)",
        data=df[at_risk],
        family=sm.families.Binomial(),
    ).fit()
    contrib = pd.Series(1.0, index=df.index)
    contrib.loc[at_risk] = (
        num.predict(df[at_risk]).to_numpy() / den.predict(df[at_risk]).to_numpy()
    )
    return contrib.groupby(df["subject"]).cumprod()


def _ipw_frame(observed: pd.DataFrame) -> pd.DataFrame:
    df = observed.sort_values(["subject", "period"]).reset_index(drop=True).copy()
    df["r"] = df["y"].notna().astype(int)
    df["y_prev"] = df.groupby("subject")["y"].shift(1)
    df["r_prev"] = df.groupby("subject")["r"].shift(1).fillna(1).astype(int)
    return df


def _weighted_gee_params(df: pd.DataFrame) -> pd.Series:
    df = df[df["r"] == 1].copy()
    df["sw"] = df.pop("sw_all").to_numpy()
    model = smf.gee(
        FORMULA,
        groups="subject",
        data=df,
        family=sm.families.Poisson(),
        cov_struct=sm.cov_struct.Independence(),
        weights=df["sw"].to_numpy(),
    )
    return model.fit(cov_type="robust", maxiter=200).params


def fit_ipw(observed: pd.DataFrame, analysis: str) -> pd.DataFrame:
    """IPW-GEE: the response-model MAR method. Point estimate from the full
    sample; SEs from a respondent-level bootstrap that re-estimates the
    weights within every replicate (the nuisance model is never held
    fixed)."""
    base = _ipw_frame(observed)
    base["sw_all"] = _stabilized_weights(base)
    point = _weighted_gee_params(base)
    if list(point.index) != list(TERMS):
        raise RuntimeError(f"unexpected IPW term ordering: {list(point.index)}")

    rng = np.random.default_rng(MI_SEED)
    subjects = base["subject"].unique()
    draws = []
    failures = 0
    while len(draws) < IPW_BOOT:
        sample = rng.choice(subjects, size=len(subjects), replace=True)
        parts = []
        for k, sid in enumerate(sample):
            block = base[base["subject"] == sid].copy()
            block["subject"] = k  # resampled participants are distinct clusters
            parts.append(block)
        boot = pd.concat(parts, ignore_index=True)
        try:
            boot["sw_all"] = _stabilized_weights(boot)
            draws.append(_weighted_gee_params(boot).to_numpy())
        except Exception:
            failures += 1
            if failures > IPW_BOOT // 4:
                raise RuntimeError("too many IPW bootstrap failures")
    se = np.std(np.stack(draws), axis=0, ddof=1)
    print(f"IPW bootstrap: {len(draws)} fits, {failures} failures")
    return pd.DataFrame(
        {
            "analysis": analysis,
            "term": list(TERMS.values()),
            "estimate": point.to_numpy(),
            "robust_se": se,
            "n_rows": int(base["r"].sum()),
            "n_subjects": observed["subject"].nunique(),
        }
    )


def write_delta_response(observed: pd.DataFrame) -> None:
    """Pooled treat:period across the prespecified delta grid."""
    rows = []
    for mult in DELTA_GRID:
        label = "mi_rubin" if mult == 1.0 else f"mi_delta_x{mult}"
        table = fit_mi(observed, label, None if mult == 1.0 else float(np.log(mult)))
        row = table[table["term"] == "treat_period"].iloc[0]
        rows.append(
            {
                "exp_delta": mult,
                "estimate": row["estimate"],
                "robust_se": row["robust_se"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "delta_response.csv", index=False)
    print(f"wrote {HERE / 'delta_response.csv'}")
    print(out.to_string(index=False))


def write_comparison_table(results: pd.DataFrame) -> None:
    """Machine-readable headline table: treat:period across methods."""
    assumptions = {
        "complete": "no induced missingness (benchmark)",
        "available_case": "response-independent observed-data analysis",
        "mi_rubin": "MAR + correct longitudinal outcome model",
        "ipw_gee": "sequential MAR + correct response model",
        "mi_under_mnar": "MAR incorrectly imposed on MNAR missingness",
        "mi_delta_low": "specified MNAR shift (means x 0.8)",
        "mi_delta_high": "specified MNAR shift (means x 1.25)",
    }
    head = results[results["term"] == "treat_period"].set_index("analysis")
    benchmark = head.loc["complete", "estimate"]
    rows = []
    for name, assumption in assumptions.items():
        if name not in head.index:
            continue
        est, se = head.loc[name, "estimate"], head.loc[name, "robust_se"]
        rows.append(
            {
                "analysis": name,
                "assumption": assumption,
                "estimate": est,
                "se": se,
                "ci_low": est - 1.959963984540054 * se,
                "ci_high": est + 1.959963984540054 * se,
                "diff_from_complete": est - benchmark,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "comparison_table.csv", index=False)
    print(f"\nwrote {HERE / 'comparison_table.csv'}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))


def main() -> None:
    epil = load_epil()
    print(f"data: {len(epil)} rows, sha256 {normalized_hash(epil)}")

    mask = pd.read_csv(mask_path("mar_monotone"))
    observed = apply_mask(epil, mask)
    available = observed.dropna(subset=["y"]).reset_index(drop=True)
    print(
        f"MAR mask: {int(mask['observed'].sum())} observed cells, "
        f"{len(available)} available-case rows"
    )

    # MNAR stress test (single realization — the systematic failure
    # demonstration lives in tests/simulation): MAR imputation applied
    # to missingness that depends on the current, hidden count
    mnar_mask = pd.read_csv(mask_path("mnar_stress_test"))
    observed_mnar = apply_mask(epil, mnar_mask)
    print(
        f"MNAR stress mask: {int(mnar_mask['observed'].sum())} observed cells"
    )

    results = pd.concat(
        [
            fit_gee(epil, "complete"),
            fit_gee(available, "available_case"),
            fit_mi(observed, "mi_rubin", delta=None),
            fit_ipw(observed, "ipw_gee"),
            fit_mi(observed_mnar, "mi_under_mnar", delta=None, cache_key="mnar"),
            # MNAR sensitivity: imputed means shifted down/up
            fit_mi(observed, "mi_delta_low", delta=float(np.log(0.8))),
            fit_mi(observed, "mi_delta_high", delta=float(np.log(1.25))),
        ],
        ignore_index=True,
    )
    out = HERE / "results_python.csv"
    results.to_csv(out, index=False)
    print(f"wrote {out}")

    write_comparison_table(results)
    write_delta_response(observed)


if __name__ == "__main__":
    main()
