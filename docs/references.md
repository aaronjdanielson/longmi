# References

The canonical citation list. Each entry notes where longmi relies on it.
longmi implements established missing-data theory; it does not introduce a
new proof of multiple imputation.

## Missing-data foundations

- Rubin DB. Inference and Missing Data. *Biometrika.* 1976;63(3):581–592.
  doi:10.1093/biomet/63.3.581.
  — *Proposition 1 (ignorability under MAR); assumptions A2–A3.*
- Little RJA, Rubin DB. *Statistical Analysis with Missing Data.* 3rd ed.
  Hoboken: Wiley; 2019. doi:10.1002/9781119482260.
  — *General reference for mechanisms and taxonomy
  ([MCAR, MAR, MNAR](explanation/mcar_mar_mnar.md)).*

## Multiple imputation and posterior-predictive draws

- Rubin DB. Multiple Imputations in Sample Surveys — A Phenomenological
  Bayesian Approach to Nonresponse. *Proceedings of the Survey Research
  Methods Section, American Statistical Association.* 1978:20–34.
  — *Origin of the MI construction (Proposition 2).*
- Rubin DB. *Multiple Imputation for Nonresponse in Surveys.* New York:
  Wiley; 1987. doi:10.1002/9780470316696.
  — *Combining rules, proper imputation, large-sample parameter draws
  (Propositions 2–3; the NB backend's declared approximation cites
  sec. 4.3).*
- Rubin DB. Multiple Imputation After 18+ Years. *Journal of the American
  Statistical Association.* 1996;91(434):473–489.
  doi:10.1080/01621459.1996.10476908.
  — *Retrospective and defense of the framework.*
- Tanner MA, Wong WH. The Calculation of Posterior Distributions by Data
  Augmentation. *Journal of the American Statistical Association.*
  1987;82(398):528–540. doi:10.1080/01621459.1987.10478458.
  — *The data-augmentation scheme behind the joint Gaussian backend.*
- Schafer JL. *Analysis of Incomplete Multivariate Data.* New York:
  Chapman & Hall/CRC; 1997.
  — *The NORM algorithm the joint Gaussian backend implements (with
  regressors).*
- White IR, Royston P, Wood AM. Multiple Imputation Using Chained
  Equations: Issues and Guidance for Practice. *Statistics in Medicine.*
  2011;30(4):377–399. doi:10.1002/sim.4067.
  — *Practical guidance informing the
  [reporting checklist](how_to/report_an_analysis.md) and choice of M.*
- Carpenter JR, Bartlett JW, Morris TP, Wood AM, Quartagno M, Kenward MG.
  *Multiple Imputation and its Application.* 2nd ed. Chichester: Wiley;
  2023. doi:10.1002/9781119756118.
  — *Modern book-length treatment, including sensitivity analysis.*

## Combining rules and degrees of freedom

- Barnard J, Rubin DB. Small-Sample Degrees of Freedom With Multiple
  Imputation. *Biometrika.* 1999;86(4):948–955.
  doi:10.1093/biomet/86.4.948.
  — *The `pool_rubin` degrees-of-freedom rule
  ([algorithm](algorithms/rubin_pooling.md)).*

## Large-sample theory, congeniality, estimating equations

- Wang N, Robins JM. Large-Sample Theory for Parametric Multiple
  Imputation Procedures. *Biometrika.* 1998;85(4):935–948.
  doi:10.1093/biomet/85.4.935. — *Proposition 3's asymptotics.*
- Meng XL. Multiple-Imputation Inferences With Uncongenial Sources of
  Input. *Statistical Science.* 1994;9(4):538–573.
  doi:10.1214/ss/1177010269. — *Congeniality (A8; Proposition 4).*
- Robins JM, Wang N. Inference for Imputation Estimators. *Biometrika.*
  2000;87(1):113–124. doi:10.1093/biomet/87.1.113.
  — *Variance under incompatibility (Proposition 4).*
- Yang S, Kim JK. A Note on Multiple Imputation for Method of Moments
  Estimation. *Biometrika.* 2016;103(1):244–251. doi:10.1093/biomet/asv073.
  — *Rubin-variance bias for method-of-moments analyses (Proposition 4).*

## GEE and marginal models

- Liang KY, Zeger SL. Longitudinal Data Analysis Using Generalized Linear
  Models. *Biometrika.* 1986;73(1):13–22. doi:10.1093/biomet/73.1.13.
  — *The GEE estimator behind `StatsmodelsGEE`.*
- Zeger SL, Liang KY, Albert PS. Models for Longitudinal Data: A
  Generalized Estimating Equation Approach. *Biometrics.*
  1988;44(4):1049–1060. doi:10.2307/2531734.
  — *Marginal vs subject-specific (conditional) estimands; the attenuation
  the NB simulation's truth calibration accounts for.*

## MI with GEE in longitudinal research

- Beunckens C, Sotto C, Molenberghs G. A Simulation Study Comparing
  Weighted Estimating Equations With Multiple Imputation Based Estimating
  Equations for Longitudinal Binary Data. *Computational Statistics & Data
  Analysis.* 2008;52(3):1533–1548. doi:10.1016/j.csda.2007.04.020.
- Lipsitz SR, Fitzmaurice GM, Weiss RD. Using Multiple Imputation With GEE
  With Non-monotone Missing Longitudinal Binary Outcomes. *Psychometrika.*
  2020;85(4):890–904. doi:10.1007/s11336-020-09729-y.

## Inverse probability weighting (the complementary approach)

- Robins JM, Rotnitzky A, Zhao LP. Analysis of Semiparametric Regression
  Models for Repeated Outcomes in the Presence of Missing Data. *Journal
  of the American Statistical Association.* 1995;90(429):106–121.
  doi:10.1080/01621459.1995.10476493.
  — *IPW-GEE, the response-model counterpart planned for the epil
  example's comparison arm.*

## Robust inference under uncongeniality

- Bartlett JW, Hughes RA. Bootstrap Inference for Multiple Imputation
  Under Uncongeniality and Misspecification. *Statistical Methods in
  Medical Research.* 2020;29(12):3533–3546. doi:10.1177/0962280220932189.
  — *Bootstrap-then-impute, the planned 0.2 pooling mode.*

## MNAR sensitivity analysis

- Cro S, Morris TP, Kenward MG, Carpenter JR. Sensitivity Analysis for
  Clinical Trials With Missing Continuous Outcome Data Using Controlled
  Multiple Imputation: A Practical Guide. *Statistics in Medicine.*
  2020;39(21):2815–2842. doi:10.1002/sim.8569.
  — *Delta adjustment as controlled MI
  ([how-to](how_to/run_delta_sensitivity.md)).*

## Numerical methods

- Liu Q, Pierce DA. A Note on Gauss–Hermite Quadrature. *Biometrika.*
  1994;81(3):624–629. doi:10.1093/biomet/81.3.624.
  — *The quadrature integrating the NB backend's random intercept.*

## Evaluating statistical methods by simulation

- Morris TP, White IR, Crowther MJ. Using Simulation Studies to Evaluate
  Statistical Methods. *Statistics in Medicine.* 2019;38(11):2074–2102.
  doi:10.1002/sim.8086.
  — *The framework behind the simulation harness's performance measures
  and Monte Carlo standard errors.*

## Applied tutorial and validation oracles

- Wijesuriya R, et al. Multiple Imputation for Longitudinal Data: A
  Tutorial. *Statistics in Medicine.* 2024. doi:10.1002/sim.10274.
  — *The external methodological oracle (simulated CATS parity).*
- Thall PF, Vail SC. Some Covariance Models for Longitudinal Count Data
  With Overdispersion. *Biometrics.* 1990;46(3):657–671.
  doi:10.2307/2532086. — *Source of the epil seizure-count data.*
- Venables WN, Ripley BD. *Modern Applied Statistics with S.* 4th ed. New
  York: Springer; 2002. doi:10.1007/978-0-387-21706-2.
  — *The MASS package distributing `epil`.*

## Software validated against or built upon

- van Buuren S, Groothuis-Oudshoorn K. mice: Multivariate Imputation by
  Chained Equations in R. *Journal of Statistical Software.*
  2011;45(3):1–67. doi:10.18637/jss.v045.i03.
  — *`pool_rubin` is verified bit-compatible with `mice::pool.scalar`.*
- Halekoh U, Højsgaard S, Yan J. The R Package geepack for Generalized
  Estimating Equations. *Journal of Statistical Software.* 2006;15(2):1–11.
  doi:10.18637/jss.v015.i02.
  — *The R side of the epil GEE cross-implementation comparison.*
- Seabold S, Perktold J. statsmodels: Econometric and Statistical Modeling
  with Python. *Proceedings of the 9th Python in Science Conference.*
  2010:92–96. doi:10.25080/Majora-92bf1922-011.
  — *The GEE/GLM engines wrapped by the analysis adapters.*
