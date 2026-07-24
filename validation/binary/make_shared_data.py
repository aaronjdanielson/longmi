"""Generate the shared binary dataset + MAR mask for the cross-method
comparison (longmi BernoulliImputer vs R mice logreg). Deterministic."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(20260725)
n = 300
treat = (rng.uniform(size=n) < 0.5).astype(float)
b = 0.8 * rng.standard_normal(n)
rows = []
for i in range(n):
    for w in (1, 2, 3):
        logit = -0.5 + 0.3*(w==2) + 0.4*(w==3) + 0.5*treat[i] + 0.7*treat[i]*(w==3) + b[i]
        rows.append((i+1, w, int(rng.uniform() < 1/(1+np.exp(-logit))), int(treat[i])))
f = pd.DataFrame(rows, columns=["pid", "wave", "y", "treat"])
y1 = f[f.wave==1].set_index("pid")["y"]
miss = (f.wave > 1) & (rng.uniform(size=len(f)) < 0.12 + 0.25*f.pid.map(y1))
f["observed"] = (~miss).astype(int)
out = Path(__file__).parent / "shared_binary.csv"
f.to_csv(out, index=False)
print(f"wrote {out}: {len(f)} rows, {int(miss.sum())} masked")
