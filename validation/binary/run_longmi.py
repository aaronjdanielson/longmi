"""longmi side: Bernoulli MI + logistic GEE + Rubin on the shared data."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from longmi import LongitudinalData, pool_rubin
from longmi.analysis import StatsmodelsGEE
from longmi.impute import BernoulliImputer

HERE = Path(__file__).parent
f = pd.read_csv(HERE / "shared_binary.csv")
f["y"] = f["y"].astype(float).where(f["observed"] == 1)
d = LongitudinalData(f, id_col="pid", time_col="wave", outcome_col="y",
                     predictor_cols=("treat",), outcome_type="binary",
                     times=(1, 2, 3))
fit = BernoulliImputer(time_interactions=("treat",)).fit(d)
cm = fit.impute(20, random_state=20260725)
gee = StatsmodelsGEE("y ~ treat * C(wave)", groups="pid",
                     family="binomial", cov_struct="exchangeable")
p = pool_rubin(cm.analyze(gee), validity=cm.declaration)
pd.DataFrame({"term": p.names, "estimate": p.qbar, "se": p.se}).to_csv(
    HERE / "results_longmi.csv", index=False)
print(pd.DataFrame({"term": p.names, "estimate": p.qbar, "se": p.se}).to_string(index=False)
      )
