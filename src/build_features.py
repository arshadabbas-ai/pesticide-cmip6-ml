"""Build the modelling table: well-sample rows + atrazine use + antecedent climate.

Climate windows use ONLY days strictly before the sample date (no leakage).
Atrazine use = county mean of EPest 2013-17 (HIGH estimate), applied to all
sample years (assumption documented in Methods); also keeps year-matched use
for the 2013-17 robustness check.
"""
import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wqp_loader import build_well_table

RAW = pathlib.Path("data/raw")
PROC = pathlib.Path("data/processed")


def county_use(pnsp_path, compound="ATRAZINE", state=19):
    p = pd.read_csv(pnsp_path, sep="\t", low_memory=False)
    a = p[(p.COMPOUND == compound) & (p.STATE_FIPS_CODE == state)].copy()
    a["county"] = a.COUNTY_FIPS_CODE.astype(int)
    a["use_kg"] = a.EPEST_HIGH_KG
    mean = a.groupby("county").use_kg.mean().rename("use_kg_mean")
    yearly = a[["county", "YEAR", "use_kg"]].rename(columns={"YEAR": "year", "use_kg": "use_kg_year"})
    return mean, yearly


def cell(lat, lon):
    return round(lat * 2) / 2, round(lon / 0.625) * 0.625


def climate_features(row, cache):
    la, lo = cell(row.lat, row.lon)
    c = cache[(la, lo)]
    end = row.date - pd.Timedelta(days=1)          # strictly before sampling
    out = {}
    for n in (30, 90, 365):
        w = c.loc[end - pd.Timedelta(days=n - 1): end]
        out[f"precip_{n}d"] = w.PRECTOTCORR.sum()
    w90 = c.loc[end - pd.Timedelta(days=89): end]
    out["temp_mean_90d"] = w90.T2M.mean()
    out["precip_max1d_90d"] = w90.PRECTOTCORR.max()
    out["heavy_rain_days_90d"] = (w90.PRECTOTCORR >= 10).sum()
    out["soilwet_90d"] = w90.GWETROOT.mean()
    return pd.Series(out)


def load_power_cache(power_dir=RAW / "power"):
    cache = {}
    for f in power_dir.glob("cell_*.csv"):
        _, la, lo = f.stem.split("_")
        cache[(float(la), float(lo))] = pd.read_csv(f, index_col="date", parse_dates=True)
    return cache


def build(fractions=("Dissolved",), with_climate=True):
    m = build_well_table(RAW / "wqp_iowa_atrazine_results.csv", RAW / "wqp_iowa_stations.csv", fractions)
    st = pd.read_csv(RAW / "wqp_iowa_stations.csv", low_memory=False)
    st = st.rename(columns={"MonitoringLocationIdentifier": "well_id", "CountyCode": "county",
                            "WellDepthMeasure/MeasureValue": "well_depth_ft"})[["well_id", "county", "well_depth_ft"]]
    m = m.merge(st.drop_duplicates("well_id"), on="well_id", how="left")
    mean, yearly = county_use(RAW / "pnsp_county_2013-17.txt")
    m = m.merge(mean, on="county", how="left").merge(yearly, on=["county", "year"], how="left")
    miss = m.use_kg_mean.isna().sum()
    if miss:
        print(f"WARNING: {miss} rows in counties with no EPest atrazine record")
    if with_climate:
        cache = load_power_cache()
        if not cache:
            raise SystemExit("No NASA POWER files found. Run: python src/download_power.py")
        m = pd.concat([m, m.apply(climate_features, axis=1, cache=cache)], axis=1)
    PROC.mkdir(parents=True, exist_ok=True)
    return m


if __name__ == "__main__":
    t = build()
    t.to_csv(PROC / "model_table_dissolved.csv", index=False)
    print(t.shape); print(t.isna().mean().round(3).to_string())
