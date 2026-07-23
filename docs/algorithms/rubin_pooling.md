# Rubin pooling — algorithm as implemented

Implementation: [src/longmi/pooling/rubin.py](../../src/longmi/pooling/rubin.py).
Theory and citations: Proposition 3 of
[mathematical_foundations.md](../theory/mathematical_foundations.md)
(Rubin 1987; Barnard & Rubin 1999).

## Inputs

$M \ge 2$ completed-data fits, each an `AnalysisEstimate` with

- parameter names, in **identical order** across fits (a permuted ordering
  raises; it is never silently realigned);
- point estimates $\widehat Q^{(m)} \in \mathbb{R}^p$;
- complete-data covariance $U^{(m)} \in \mathbb{R}^{p \times p}$ (validated
  symmetric with nonnegative diagonal);
- a common complete-data degrees of freedom $\nu_{\mathrm{com}}$, or `None`
  for the large-sample reference.

## Formulas

$$
\overline Q = \frac{1}{M}\sum_m \widehat Q^{(m)},
\qquad
\overline U = \frac{1}{M}\sum_m U^{(m)},
$$

$$
B = \frac{1}{M-1}\sum_m
  (\widehat Q^{(m)} - \overline Q)(\widehat Q^{(m)} - \overline Q)^\top,
\qquad
T = \overline U + \left(1 + \frac{1}{M}\right) B.
$$

Per-parameter quantities use the diagonals ($j = 1, \dots, p$), matching
`mice::pool.scalar` (rule `"rubin1987"`):

- relative increase in variance:
  $r_j = (1 + 1/M)\, B_{jj} / \overline U_{jj}$;
- proportion of total variance due to missingness:
  $\lambda_j = (1 + 1/M)\, B_{jj} / T_{jj}$;
- Rubin (1987) large-sample degrees of freedom:
  $\nu_{\mathrm{old},j} = (M - 1) / \lambda_j^2$
  (infinite when $\lambda_j = 0$);
- Barnard–Rubin (1999) small-sample adjustment, applied when
  $\nu_{\mathrm{com}}$ is finite:

$$
\nu_{\mathrm{obs},j}
= \frac{\nu_{\mathrm{com}} + 1}{\nu_{\mathrm{com}} + 3}\,
  \nu_{\mathrm{com}} (1 - \lambda_j),
\qquad
\nu_j = \frac{\nu_{\mathrm{old},j}\, \nu_{\mathrm{obs},j}}
             {\nu_{\mathrm{old},j} + \nu_{\mathrm{obs},j}};
$$

- fraction of missing information:
  $\gamma_j = \dfrac{r_j + 2/(\nu_j + 3)}{1 + r_j}$
  (with $2/(\nu_j+3) \to 0$ when $\nu_j = \infty$);
- confidence intervals:
  $\overline Q_j \pm t_{\nu_j,\,1-\alpha/2}\sqrt{T_{jj}}$, with the normal
  reference when $\nu_j = \infty$.

## Edge cases

- $B_{jj} = 0$: $r_j = \lambda_j = \gamma_j = 0$, $\nu_j = \infty$,
  $T_{jj} = \overline U_{jj}$.
- $M = 1$: refused — between-imputation variance is undefined.
- Non-positive $\overline U_{jj}$: refused.
- Inconsistent `dfcom` across fits: refused.

## Deviation from mice

`mice` clamps $\lambda < 10^{-4}$ up to $10^{-4}$ inside its Barnard–Rubin
computation; `longmi` instead propagates the exact value (with the
$\lambda = 0$ limit handled analytically). Agreement with `mice` is
therefore exact for $\lambda \ge 10^{-4}$ — any realistic amount of missing
information — which the cross-language suite verifies at relative tolerance
$10^{-12}$ ([validation/r/rubin_reference.R](../../validation/r/rubin_reference.R),
[tests/cross_language/](../../tests/cross_language/)).
