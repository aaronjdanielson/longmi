# Project status

The canonical feature-maturity table. Other pages (README, examples,
algorithm docs) should defer to this file rather than restate status;
detailed exact-vs-approximate computation claims live only in the
algorithm pages.

Version: `0.1.0a3` (see RELEASE_NOTES.md). Public API is **experimental** —
expected to stabilize after the simulation grid below is complete.

## Feature maturity

| Feature | Implementation | Unit tests | Numerical diagnostics | R parity | Simulation bias/coverage | API status |
| --- | --- | --- | --- | --- | --- | --- |
| `LongitudinalData` / completion invariants | complete | complete | n/a | n/a | n/a | experimental |
| `pool_rubin` (Rubin + Barnard–Rubin) | complete | complete | n/a | **complete** — bit-compatible with `mice::pool.scalar`, verified at 1e-12 incl. boundary cases | n/a | experimental |
| `JointGaussianImputer` | complete | complete | chain autocorrelation/ESS per run | pending | **initial matrix passing** (MCAR, monotone & intermittent MAR, expected-failure demos; see below) | experimental |
| `NegativeBinomialImputer` | complete | complete | optimizer, curvature, quadrature, grid boundary mass | pending | **initial matrix passing** (monotone & intermittent MAR, high-missingness stress, expected-failure demos, delta curve; see below) | experimental |
| `StatsmodelsGEE` adapter | complete | complete — exact agreement with direct statsmodels fits, incl. the motivating registry analysis (private data, external harness) | n/a | n/a | covered via NB study | experimental |
| `StatsmodelsGLM` adapter | complete | complete | n/a | n/a | pending | experimental |
| Delta-adjusted MNAR sensitivity | complete (scalar delta; group/wave-specific deltas planned) | complete | n/a | pending | pending | experimental |
| `BernoulliImputer` (binary outcomes) | complete | complete | optimizer, curvature, separation, grid | pending | smoke matrix (MCAR/MAR validated vs marginal GEE target; omitted-interaction & MNAR expected failures; delta curve) | experimental |
| Targeted delta rules (`where`/`times`, `DeltaScenario`) | complete (wired in Bernoulli backend) | complete | n/a | pending | via delta curve | experimental |
| Bootstrap-then-impute pooling | planned | — | — | — | — | — |
| Eligibility indicators (structural missingness) | planned | — | — | — | — | — |

## Simulation evidence to date

Seeded studies in `tests/simulation/` (`pytest -m slow`) with
pre-specified, Monte-Carlo-aware gates. **Release-scale run archived at
500 replicates/scenario** in `validation/releases/0.1.0a1/`
(simulation_summary.md is the authoritative table; source commit
recorded there). Standardized bias = bias / empirical SD; SE ratio =
mean reported SE / empirical SD. Headline 500-rep results:

**Validated scenarios** (correct model; gates: low failure rate,
unbiasedness, nominal coverage, SE calibration):

| Study | Std. bias | Coverage | SE ratio | Mean FMI | Failures |
| --- | --- | --- | --- | --- | --- |
| Gaussian, MCAR 25%, wave-3 treatment effect | −0.07 | 0.948 | 0.98 | 0.24 | 0/500 |
| Gaussian, monotone MAR, same target | +0.01 | 0.944 | 0.96 | 0.35 | 0/500 |
| Gaussian, intermittent (non-monotone) MAR | −0.08 | 0.964 | 1.03 | 0.22 | 0/500 |
| NB + Poisson GEE + Rubin, monotone MAR, treat×wave-3 | +0.01 | 0.960 | 1.03 | 0.39 | 0/500 |
| NB + GEE, intermittent MAR | −0.08 | 0.948 | 1.00 | 0.25 | 0/500 |
| NB + GEE, high-missingness stress (ORR-like attrition) | +0.11 | 0.942 | 0.94 | **0.63** | 4/500 (0.8%) |

(MCSE of coverage at S=500 is ~0.0097.)

**Expected-failure demonstrations** (one assumption deliberately violated;
gates: the violation must visibly bite):

| Study | Violated | Std. bias | Coverage | Note |
| --- | --- | --- | --- | --- |
| NB imputation omitting the exposure-by-time interaction | A8 | **−1.48** (attenuation) | 0.950 | bias is masked by conservative Rubin SEs (SE ratio 1.63) — the Meng (1994) uncongeniality phenomenon in action |
| MNAR generation analyzed under MAR (NB, wave-3 mean) | A2 | **−7.38** | **0.018** | MAR MI does not rescue MNAR |
| Gaussian imputation omitting a covariate driving outcome and dropout | A5/A8 | **−1.10** | 0.798 | conditioning on too little turns MAR into effective MNAR |

**Delta-response curve** (NB, shared randomness across scenarios): the
pooled wave-3 mean is strictly increasing in delta
(×0.70 → 6.28, MAR → 7.92, ×1.40 → 9.66), brackets the MAR result, and its
spread matches the imputed share of wave-3 cells.

Not yet demonstrated: wider grids over sample size / missingness fraction
/ correlation / overdispersion, misspecified dependence structures, R
parity for the imputation backends.
**The backends should not be described as statistically validated beyond
this table.**

## Cross-language validation

- Rubin pooling: exact parity with `mice` (complete).
- epil example: Python (statsmodels) vs R (geepack) complete-data and
  available-case GEE agree within 2e-3 (cross-implementation, numerical
  agreement); the MI arm is additionally compared cross-method against R
  (mice wide-PMM + geepack + mice::pool on the same mask) — *statistical
  agreement*: pooled estimates within 0.5 pooled SEs, SE ratios 0.88-1.21.
  The full narrative (complete / available-case / MI / IPW / MNAR failure
  / delta curve) is implemented; see examples/epil_count.
- CATS tutorial parity: external checkout pinned and verified; the
  statistical comparison is in progress.
