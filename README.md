# Back on the Rails: Tracking DC’s Metro Revival

This repository contains the analysis and Tableau dashboard build for DC Metrorail ridership recovery.

## Quick Start
1. Create a virtual environment and install requirements:
   ```bash
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   ```bash
   # Windows
   python -m venv .venv  
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Place WMATA ridership CSV exports here: `data_raw/`
   - Example filenames: `ridership_2019.csv`, `ridership_2023.csv`, `ridership_2024.csv`, `ridership_2025.csv`

3. Place DC Open Data geometry here: `data_geo/`
   - `metro_stations.geojson` (Metro Stations - Regional)
   - `metro_lines.geojson` (Metro Lines - Regional)

4. Run the scripts in order:
   ```bash
   python analysis/01_clean_data.py
   python analysis/02_inference_tests.py
   python analysis/03_tableau_export.py
   ```

5. Connect `data_clean/tableau_recovery.csv` in Tableau to build the dashboard.

## Data Sources
- WMATA Ridership Portal: https://www.wmata.com/initiatives/ridership-portal/Metrorail-Ridership-Summary.cfm
- DC Open Data – Metro Stations (Regional): https://opendata.dc.gov/datasets/DCGIS::metro-stations-regional/about
- DC Open Data – Metro Lines (Regional): https://opendata.dc.gov/datasets/DCGIS::metro-lines-regional/about

## Notes
- Recovery Ratio = current entries / 2019 baseline entries (aligned by time period).
- See WMATA metadata for notes and other caveats.
