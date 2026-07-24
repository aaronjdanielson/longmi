"""Cross-method comparison gate: longmi Bernoulli MI vs R mice(logreg).
Run make_shared_data.py, run_longmi.py, run_mice.R first. Exit 1 on
disagreement beyond statistical tolerance."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent
lm = pd.read_csv(HERE / "results_longmi.csv").rename(
    columns={"se": "se_longmi", "estimate": "est_longmi"})
mc = pd.read_csv(HERE / "results_mice.csv").rename(
    columns={"se": "se_mice", "estimate": "est_mice"})
name_map = {"Intercept": "(Intercept)", "treat": "treat",
            "C(wave)[T.2]": "factor(wave)2", "C(wave)[T.3]": "factor(wave)3",
            "treat:C(wave)[T.2]": "treat:factor(wave)2",
            "treat:C(wave)[T.3]": "treat:factor(wave)3"}
lm["rterm"] = lm["term"].map(name_map)
cmp = lm.merge(mc, left_on="rterm", right_on="term")
cmp["diff_in_se"] = (cmp.est_longmi - cmp.est_mice).abs() / np.maximum(
    cmp.se_longmi, cmp.se_mice)
cmp["se_ratio"] = cmp.se_longmi / cmp.se_mice
print(cmp[["rterm", "est_longmi", "est_mice", "diff_in_se",
           "se_ratio"]].round(3).to_string(index=False))
cmp.to_csv(HERE / "comparison.csv", index=False)
ok = (cmp.diff_in_se < 1.0).all() and cmp.se_ratio.between(0.5, 2).all()
print("\nOK: cross-method statistical agreement" if ok else "\nFAIL")
sys.exit(0 if ok else 1)
