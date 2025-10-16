import os
import pandas as pd
import geopandas as gpd

IN = "data_clean/station_monthly_recovery.csv"
STATIONS = "data_geo/metro_stations.geojson"
OUT = "data_clean/tableau_recovery.csv"

os.makedirs("data_clean", exist_ok=True)

# 1) Load data
df = pd.read_csv(IN)
df["Station"] = df["Station"].astype(str).str.upper().str.strip()

g = gpd.read_file(STATIONS)

# 2) Normalize station name column from DC Open Data
name_candidates = [c for c in ["NAME","STATION","STATION_NAME","NAME_LONG","NAME1"] if c in g.columns]
if not name_candidates:
    name_candidates = [c for c in g.columns if c != "geometry"]
name_col = name_candidates[0]
g[name_col] = g[name_col].astype(str).str.upper().str.strip()

# 3) Ensure source CRS is set; DC Open Data GeoJSON is typically EPSG:4326
if g.crs is None:
    # Assume WGS84 if missing
    g = g.set_crs(epsg=4326)

# 4) Reproject to a projected CRS for centroid computation (UTM 18N covers DC)
g_proj = g.to_crs(epsg=32618)

# 5) Compute a robust point for each feature
#    For points: centroid == the point; for lines/polygons, representative_point() avoids odd edge cases
geom_centroid = g_proj.geometry.centroid
geom_repr = g_proj.geometry.representative_point()

# Prefer centroid where valid; fall back to representative point if centroid is empty/null
centers = geom_centroid.copy()
centers[centers.is_empty | centers.isna()] = geom_repr[centers.is_empty | centers.isna()]

# 6) Convert the centroid points back to WGS84 for lon/lat
centers_wgs84 = gpd.GeoSeries(centers, crs=g_proj.crs).to_crs(epsg=4326)

# 7) Build a minimal stations table with lon/lat
g_small = g[[name_col]].copy()
g_small["Longitude"] = centers_wgs84.x.values
g_small["Latitude"] = centers_wgs84.y.values
g_small = g_small.drop_duplicates()
g_small.columns = ["Station", "Longitude", "Latitude"]

# 8) Join to ridership table and export for Tableau
out = df.merge(g_small, on="Station", how="left")
out.to_csv(OUT, index=False)
print(f"Saved {OUT} with {len(out):,} rows.")
