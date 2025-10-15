
import pandas as pd
import geopandas as gpd
import os

IN = "data_clean/station_monthly_recovery.csv"
STATIONS = "data_geo/metro_stations.geojson"
OUT = "data_clean/tableau_recovery.csv"

os.makedirs("data_clean", exist_ok=True)

df = pd.read_csv(IN)
g = gpd.read_file(STATIONS)

# Normalize join keys
df["Station"] = df["Station"].str.upper().str.strip()
name_col = "NAME" if "NAME" in g.columns else g.columns[0]
g[name_col] = g[name_col].astype(str).str.upper().str.strip()

# Get coordinates from geometry
g["Longitude"] = g.geometry.centroid.x
g["Latitude"] = g.geometry.centroid.y

# Reduce geometry columns
g_small = g[[name_col, "Longitude", "Latitude"]].drop_duplicates()
g_small.columns = ["Station", "Longitude", "Latitude"]

out = df.merge(g_small, on="Station", how="left")

out.to_csv(OUT, index=False)
print(f"✅ Tableau feed saved to {OUT} with {len(out):,} rows.")
