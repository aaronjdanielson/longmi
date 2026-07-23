# Simulation tests (planned)

Frequentist performance under known data-generating processes, scheduled
with the first imputation backends:

- negligible bias under MCAR;
- negligible bias under correctly modeled MAR;
- near-nominal confidence-interval coverage;
- appropriate widening with increasing missing information;
- demonstrated failure under deliberately misspecified imputation models;
- expected movement under MNAR delta shifts.

Simulation grids will live here as reproducible, seeded pytest modules,
marked slow and excluded from the default test run.
