"""Stress tests for the baseline result (run after train_baseline.py).

A. Feature-group ablation under county-held-out CV: use-only, climate-only,
   climate+use, +year, +location, and the full set.
B. Leave-years-out CV: train on some sampling years, test on others (grouped by year) -
   checks that results are not driven by lab/method changes over time.
C. Permutation importance of each feature on the full model (county-held-out).
Run:  python src/ablation.py | Tee-Object -FilePath outputs\ablation_results.txt
"""
import pathlib, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

df = pd.read_csv("data/processed/model_table_dissolved.csv")
Y = "detect_005"
CLIM = ["precip_30d", "precip_90d", "precip_365d", "temp_mean_90d",
        "precip_max1d_90d", "heavy_rain_days_90d", "soilwet_90d"]
USE = ["use_kg_mean"]
SETS = {
    "use only": USE,
    "well depth only": ["well_depth_ft"],
    "year only": ["year"],
    "climate only": CLIM,
    "climate + use": CLIM + USE,
    "climate + use + depth": CLIM + USE + ["well_depth_ft"],
    "climate + use + depth + year": CLIM + USE + ["well_depth_ft", "year"],
    "climate + use + depth + lat/lon": CLIM + USE + ["well_depth_ft", "lat", "lon"],
}


def model():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss", n_jobs=2)


def cv_scores(cols, group, seeds=range(5), splitter="sgkf"):
    out = []
    for s in seeds:
        sp = (StratifiedGroupKFold(5, shuffle=True, random_state=s) if splitter == "sgkf"
              else GroupKFold(5))
        for tr, te in sp.split(df, df[Y], df[group]):
            if df[Y].iloc[te].nunique() < 2:
                continue
            m = model().fit(df[cols].iloc[tr], df[Y].iloc[tr])
            p = m.predict_proba(df[cols].iloc[te])[:, 1]
            out.append((roc_auc_score(df[Y].iloc[te], p), average_precision_score(df[Y].iloc[te], p)))
        if splitter != "sgkf":
            break
    a = np.array(out)
    return f"AUC {a[:,0].mean():.3f} +/- {a[:,0].std():.3f}   AUPR {a[:,1].mean():.3f}"


print("A. Feature-group ablation, county-held-out CV")
for k, c in SETS.items():
    print(f"  {k:34s} {cv_scores(c, 'county')}")

print("\nB. Leave-years-out CV (grouped by sampling year), no randomness")
for k in ("climate only", "climate + use", "climate + use + depth"):
    print(f"  {k:34s} {cv_scores(SETS[k], 'year', splitter='gkf')}")

print("\nC. Permutation importance (AUC drop), full model, county-held-out")
cols = SETS["climate + use + depth"]
drops = {c: [] for c in cols}
rng = np.random.default_rng(0)
for s in range(3):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(df, df[Y], df["county"]):
        if df[Y].iloc[te].nunique() < 2:
            continue
        m = model().fit(df[cols].iloc[tr], df[Y].iloc[tr])
        Xte, yte = df[cols].iloc[te].copy(), df[Y].iloc[te]
        base = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        for c in cols:
            X2 = Xte.copy(); X2[c] = rng.permutation(X2[c].values)
            drops[c].append(base - roc_auc_score(yte, m.predict_proba(X2)[:, 1]))
for c, v in sorted(drops.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"  {c:24s} {np.mean(v):+.3f}")
