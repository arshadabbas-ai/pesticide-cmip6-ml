"""Load and clean Water Quality Portal (WQP) groundwater atrazine data.

Inputs (saved by you in data/raw/):
  wqp_iowa_atrazine_results.csv   (Result/search, dataProfile=narrowResult)
  wqp_iowa_stations.csv           (Station/search)

Design decisions (documented for the Methods section):
  * Primary subset = Dissolved fraction (most homogeneous lab method, NWIS/NAWQA).
  * Total fraction kept for a robustness check; Recoverable dropped by default (2000-2006 only).
  * ng/L converted to ug/L.
  * Non-detects: blank value + "Not Detected" -> detect = 0.
  * Detection label: any_detect (value>0 or 'Detected Not Quantified') and
    detect_005 (value >= 0.05 ug/L, a common reporting-level-harmonised threshold).
  * Detection limits are NOT in the narrow profile; the 0.05 threshold mitigates
    lab-to-lab detection-limit differences.
"""
import pandas as pd
import numpy as np

STATION_COLS = {
    "id": "MonitoringLocationIdentifier",
    "lat": "LatitudeMeasure",
    "lon": "LongitudeMeasure",
}
UNIT_TO_UGL = {"ug/l": 1.0, "ng/l": 0.001, "mg/l": 1000.0}


def load_results(path, fractions=("Dissolved",)):
    d = pd.read_csv(path, low_memory=False)
    d = d[d["CharacteristicName"].str.lower() == "atrazine"].copy()
    if fractions is not None:
        d = d[d["ResultSampleFractionText"].isin(fractions)]
    d["date"] = pd.to_datetime(d["ActivityStartDate"], errors="coerce")
    d["value_raw"] = pd.to_numeric(d["ResultMeasureValue"], errors="coerce")
    unit = d["ResultMeasure/MeasureUnitCode"].str.lower()
    factor = unit.map(UNIT_TO_UGL)
    d["value_ugl"] = d["value_raw"] * factor
    cond = d["ResultDetectionConditionText"].fillna("")
    not_detected = cond.eq("Not Detected")
    detected_nq = cond.eq("Detected Not Quantified")
    has_value = d["value_ugl"].notna()
    # drop rows that are neither a usable number nor a recognised non-detect
    keep = has_value | not_detected | detected_nq
    d = d[keep].copy()
    d.loc[not_detected.reindex(d.index), "value_ugl"] = 0.0
    d["any_detect"] = ((d["value_ugl"] > 0) | detected_nq.reindex(d.index)).astype(int)
    d["detect_005"] = (d["value_ugl"] >= 0.05).astype(int)
    d["year"] = d["date"].dt.year
    d = d.rename(columns={"MonitoringLocationIdentifier": "well_id",
                          "ResultSampleFractionText": "fraction"})
    return d[["well_id", "date", "year", "fraction", "value_ugl",
              "any_detect", "detect_005", "ProviderName"]].reset_index(drop=True)


def load_stations(path):
    s = pd.read_csv(path, low_memory=False)
    missing = [c for c in STATION_COLS.values() if c not in s.columns]
    if missing:
        raise KeyError(f"Stations file lacks {missing}. Columns found: {list(s.columns)[:30]}")
    s = s.rename(columns={v: k for k, v in STATION_COLS.items()})
    s = s[["id", "lat", "lon"]].drop_duplicates("id").dropna()
    return s.rename(columns={"id": "well_id"})


def build_well_table(results_path, stations_path, fractions=("Dissolved",)):
    r = load_results(results_path, fractions)
    s = load_stations(stations_path)
    m = r.merge(s, on="well_id", how="left", validate="m:1")
    n_missing = m["lat"].isna().sum()
    if n_missing:
        print(f"WARNING: {n_missing} rows lack coordinates; dropping.")
    return m.dropna(subset=["lat", "lon"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    r = load_results(sys.argv[1], fractions=("Dissolved",))
    print(r.shape, r["well_id"].nunique(), "wells")
    print(r[["any_detect", "detect_005"]].mean())
    print(r["year"].min(), r["year"].max())
