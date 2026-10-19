"""First real-data models: does adding atrazine use + antecedent climate beat simple baselines?

Target: detect_005 (atrazine >= 0.05 ug/L), Dissolved fraction, Iowa wells 2000-2017.
CV: grouped by WELL (no well appears in train and test) and, stricter, grouped by
COUNTY (tests generalisation to unseen counties). Repeated over seeds.
Models (feature sets):
  M0 prior       : predicts the training prevalence (sanity floor)
  M1 location    : lat, lon only (spatial baseline - shows how much is just geography)
  M2 climate     : antecedent climate only (the 'climatology' baseline)
  M3 use+climate : county atrazine use + climate + well depth (the full historical model)
Run:  python src/train_baseline.py
"""
import sys, pathlib, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

TABLE = pathlib.Path("data/processed/model_table_dissolved.csv")
TARGET = "detect_005"
CLIM = ["precip_30d", "precip_90d", "precip_365d", "temp_mean_90d",
        "precip_max1d_90d", "heavy_rain_days_90d", "soilwet_90d"]
SETS = {
    "M1 location": ["lat", "lon"],
    "M2 climate": CLIM,
    "M3 use+climate": CLIM + ["use_kg_mean", "well_depth_ft"],
}


def xgb():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss", n_jobs=2)


def run(df, group_col, seeds=range(5), n_splits=5):
    rows = []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in cv.split(df, df[TARGET], df[group_col]):
            ytr, yte = df[TARGET].iloc[tr], df[TARGET].iloc[te]
            if yte.nunique() < 2:
                continue
            res = {"M0 prior": np.full(len(te), ytr.mean())}
            for name, cols in SETS.items():
                m = xgb(); m.fit(df[cols].iloc[tr], ytr)
                res[name] = m.predict_proba(df[cols].iloc[te])[:, 1]
            lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(max_iter=1000))
            lr.fit(df[SETS["M3 use+climate"]].iloc[tr], ytr)
            res["M3b logistic"] = lr.predict_proba(df[SETS["M3 use+climate"]].iloc[te])[:, 1]
            for name, p in res.items():
                rows.append(dict(model=name, seed=seed, auc=roc_auc_score(yte, p),
                                 aupr=average_precision_score(yte, p),
                                 f1=f1_score(yte, p >= 0.5)))
    r = pd.DataFrame(rows)
    return r.groupby("model")[["auc", "aupr", "f1"]].agg(["mean", "std"]).round(3)


if __name__ == "__main__":
    df = pd.read_csv(TABLE)
    print(f"{len(df)} rows, {df.well_id.nunique()} wells, {df.county.nunique()} counties; "
          f"prevalence of {TARGET} = {df[TARGET].mean():.3f}")
    for g in ("well_id", "county"):
        print(f"\n=== Grouped CV by {g} ===")
        print(run(df, g).to_string())
