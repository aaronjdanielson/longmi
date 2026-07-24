# Release validation plan — 0.1.0a1

Pre-specified before the release run; see tests/simulation/README.md for
the scenario matrix and harness.py for the gate definitions.

- Replicates: LONGMI_SIM_REPS=500 for every scenario (uniform count for a
  simple report; MCSE of 95% coverage at S=500 is ~0.0097).
- Imputations per replicate: M=10 (Gaussian), M=8 (NB) as in the studies.
- Estimands, DGP parameters, missingness mechanisms, analysis models, and
  seeds: exactly as coded in tests/simulation/*.py at the release commit.
- Acceptance gates (pre-specified in harness.py): standardized bias gate
  0.10 (validated) / 0.25 minimum (expected failure), nominal-SE coverage
  band, SE ratio in [0.80, 1.30], numerical failure rate <= 1% at release
  scale.
- Outcome vocabulary: validated in evaluated scenarios / expected failure
  under violated assumptions / numerically unstable / not yet evaluated.

To reproduce: `LONGMI_SIM_REPS=500 pytest -m slow tests/simulation -q -s`
at the code commit recorded in source_commit.txt (the archive itself is
committed afterward — a commit cannot contain its own SHA); raw pytest
output is archived in
simulation_output.txt alongside environment_python.txt.
