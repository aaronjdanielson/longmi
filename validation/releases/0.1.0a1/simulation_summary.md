# Release-scale simulation summary — 0.1.0a1

500 replicates per scenario; source commit in source_commit.txt; plan in
simulation_plan.md; raw output in simulation_output.txt; machine-readable
results in simulation_results.csv. All metrics are conditional on the
replicates that fit successfully; the numerical-failure rate is reported
separately (release gate <= 1%, met by every validated scenario;
MNAR-under-MAR's 2.0% is within its expected-failure tolerance). MCSE of
coverage at S=500, c=0.95 is ~0.0097. The delta-response curve is
monotone, brackets MAR, and its magnitude is consistent with the imputed
share (see simulation_output.txt).

| Study | Std. bias | Coverage | SE ratio | Mean FMI | Failures |
| --- | --- | --- | --- | --- | --- |
| negbin omitted-interaction | -1.483 | 0.950 | 1.631 | 0.345 | 0/500 (0.0%) |
| negbin MNAR-under-MAR | -7.381 | 0.018 | 1.158 | 0.723 | 10/500 (2.0%) |
| gaussian omitted-auxiliary | -1.100 | 0.798 | 0.995 | 0.419 | 0/500 (0.0%) |
| gaussian mcar | -0.068 | 0.948 | 0.983 | 0.243 | 0/500 (0.0%) |
| gaussian mar | +0.013 | 0.944 | 0.958 | 0.345 | 0/500 (0.0%) |
| gaussian intermittent mar | -0.081 | 0.964 | 1.029 | 0.216 | 0/500 (0.0%) |
| negbin mar | +0.005 | 0.960 | 1.030 | 0.387 | 0/500 (0.0%) |
| negbin intermittent mar | -0.075 | 0.948 | 1.003 | 0.254 | 0/500 (0.0%) |
| negbin high-missingness | +0.106 | 0.942 | 0.937 | 0.627 | 4/500 (0.8%) |

Classification (pre-specified vocabulary):

- **Validated in evaluated scenarios** — gaussian mcar / mar /
  intermittent mar; negbin mar / intermittent mar; negbin
  high-missingness (std. bias +0.106 sits marginally above the 0.10
  target but within the MC-aware gate, 4 x MCSE, at S=500; coverage
  0.942 with FMI 0.63).
- **Expected statistical failure under violated assumptions** — omitted
  interaction: substantial bias masked by conservative Rubin variance
  under incompatibility (SE ratio 1.63) — nominal coverage alone is not
  evidence of a good estimator; MNAR-under-MAR: bias -7.4 SD, coverage
  0.018; omitted auxiliary: bias -1.1 SD, coverage 0.798.
- **Numerically unstable** — none.
- **Not yet evaluated** — wider grids over n / missingness fraction /
  correlation / overdispersion; misspecified dependence; backend-level R
  parity.
