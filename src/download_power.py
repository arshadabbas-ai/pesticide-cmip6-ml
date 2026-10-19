"""Download NASA POWER daily climate once and cache it locally (free, no API key).

Run from the project root:   python src/download_power.py
Wells are snapped to the POWER grid (0.5 deg lat x 0.625 deg lon), so Iowa needs
only ~100 requests instead of 543. Already-downloaded cells are skipped, so you
can stop and re-run safely.
Output: data/raw/power/cell_<lat>_<lon>.csv  (git-ignored with data/raw/)
"""
import sys, time, pathlib
import pandas as pd, requests
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wqp_loader import load_stations

RAW = pathlib.Path("data/raw")
OUT = RAW / "power"
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,GWETROOT"  # precip mm/d, temps C, root-zone soil wetness (0-1)
START, END = "19990101", "20171231"


def cell(lat, lon):
    return round(lat * 2) / 2, round(lon / 0.625) * 0.625


def fetch(lat, lon):
    r = requests.get(URL, params=dict(parameters=PARAMS, community="AG", latitude=lat,
                     longitude=lon, start=START, end=END, format="JSON"), timeout=120)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["properties"]["parameter"])
    d.index = pd.to_datetime(d.index, format="%Y%m%d")
    return d.replace(-999, float("nan"))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_stations(RAW / "wqp_iowa_stations.csv")
    cells = sorted({cell(a, b) for a, b in zip(s.lat, s.lon)})
    print(len(cells), "grid cells to fetch")
    for i, (la, lo) in enumerate(cells, 1):
        f = OUT / f"cell_{la:.3f}_{lo:.3f}.csv"
        if f.exists():
            continue
        for attempt in range(3):
            try:
                fetch(la, lo).to_csv(f, index_label="date")
                print(f"[{i}/{len(cells)}] saved {f.name}")
                break
            except Exception as e:
                print("retry", attempt + 1, e); time.sleep(5 * (attempt + 1))
        time.sleep(1)
    print("done")
