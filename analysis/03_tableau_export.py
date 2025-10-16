
import pandas as pd
import geopandas as gpd
import os

IN = "data_clean/station_monthly_recovery.csv"
STATIONS = "data_geo/metro_stations.geojson"
OUT = "data_clean/tableau_recovery.csv"

os.makedirs("data_clean", exist_ok=True)

df = pd.read_csv(IN)
g = gpd.read_file(STATIONS)

df["Station"] = df["Station"].astype(str).str.upper().str.strip()

# Choose name column
name_candidates = [c for c in ["NAME","STATION","STATION_NAME","NAME_LONG","NAME1"] if c in g.columns]
if not name_candidates:
    name_candidates = [c for c in g.columns if c != "geometry"]
name_col = name_candidates[0]

g[name_col] = g[name_col].astype(str).str.upper().str.strip()

g = g.to_crs(4326)
g["Longitude"] = g.geometry.centroid.x
g["Latitude"] = g.geometry.centroid.y

g_small = g[[name_col, "Longitude", "Latitude"]].drop_duplicates()
g_small.columns = ["Station", "Longitude", "Latitude"]

out = df.merge(g_small, on="Station", how="left")

out.to_csv(OUT, index=False)
print(f"Saved {OUT} with {len(out):,} rows.")
