# NWR Coverage Gap Research

**Researcher:** Andrew Cooper  
**Mentor:** Nick Langan, Villanova University  
**Goal:** Identify underserved populations in the NOAA Weather Radio All Hazards (NWR) network and compile a formal gap report suitable for submission to FEMA.

---

## Research Plan

| Week | Focus |
|------|-------|
| 1 | Get to know the NWR system and the station dataset |
| 2 | Identify datasets for gap analysis — hazards, population, demographics |
| 3 | Evaluate signal coverage using RadioLand API endpoints (terrain, power, HAGL) |
| 4–5 | Data visualization — map coverage holes across the US |
| 6–7 | Rank the top 50 most at-risk areas; begin formal report |
| 8 | Finalize report with Nick |

---

## Dataset

`wx_stations.csv` — 1,033 NWR transmitters sourced from FCC/NOAA data via RadioLand.

**Key columns:**

| Column | Description |
|--------|-------------|
| `callsign` | Station identifier (e.g., `WXL58`) |
| `frequency` | One of 7 NWR frequencies (162.400–162.550 MHz) |
| `city` / `state_province` / `country` | Transmitter location |
| `erp_watts` | Effective Radiated Power — higher = broader coverage |
| `latitude` / `longitude` | Transmitter coordinates (longitude stored as positive — negate for correct US mapping) |
| `wfo` | Assigned NOAA Weather Forecast Office |
| `transmitter_site` | Tower site name when available |
| `hagl` | Height Above Ground Level in feet (antenna height) |

**To filter to US-only stations in Python:**
```python
wx = wx[wx['country'] == 'USA'].dropna(subset=['latitude', 'longitude'])
wx['longitude'] = wx['longitude'] * -1
```

---

## Materials

- [`week1_nwr_orientation.ipynb`](week1_nwr_orientation.ipynb) — Week 1: learn the system, explore the data
- [`week2_datasets_and_gaps.ipynb`](week2_datasets_and_gaps.ipynb) — Week 2: hazard taxonomy, demographic datasets, research framing
- [`week3_coverage_modeling.ipynb`](week3_coverage_modeling.ipynb) — Week 3: Longley-Rice signal modeling via RadioLand API
- [`week4_5_visualization.ipynb`](week4_5_visualization.ipynb) — Weeks 4–5: gap maps, coverage holes across the US

---

## Hazard Overlay Datasets

Four hazard datasets are in [`data/`](data/) for use in the Week 6–7 gap-ranking and FEMA report. All files have valid `lat`/`lon` columns for direct mapping.

### `data/coastal_flood_events.csv` — 4,591 events (1996–2025)
Source: [NOAA Storm Events Database](https://www.ncdc.noaa.gov/stormevents/)

| Column | Description |
|--------|-------------|
| `year` / `month` | Event date |
| `state` / `county` | Location name |
| `cz_type` | `C` = county-level report, `Z` = forecast zone report |
| `wfo` | Issuing NWS Weather Forecast Office |
| `lat` / `lon` | County centroid (CZ_TYPE=C) or WFO office location (CZ_TYPE=Z) |
| `deaths` | Direct fatalities |
| `damage` | Property damage string (e.g. `10.00K`) |
| `narrative` | Event description (truncated to 200 chars) |

### `data/wildfire_events.csv` — 8,440 events (1996–2025)
Source: [NOAA Storm Events Database](https://www.ncdc.noaa.gov/stormevents/)  
Same columns as `coastal_flood_events.csv`.

### `data/us_hurricane_landfalls.csv` — 607 landfalls (1935–2024)
Source: [IBTrACS v04r01](https://www.ncei.noaa.gov/products/international-best-track-archive) — USA_RECORD=`L` landfall points only

| Column | Description |
|--------|-------------|
| `year` | Season year |
| `name` | Storm name |
| `iso_time` | UTC timestamp of landfall |
| `lat` / `lon` | Landfall coordinates |
| `sshs` | Saffir-Simpson category at landfall (-1 = tropical storm/depression) |
| `wind_kt` | Sustained wind speed in knots |

### `data/tornadoes_f3plus.csv` — 3,278 tornadoes (1950–2024)
Source: [SPC Severe Weather Database](https://www.spc.noaa.gov/wcm/#data) — F3/EF3 and above, CONUS only

| Column | Description |
|--------|-------------|
| `yr` / `mo` / `dy` | Date |
| `st` | State abbreviation |
| `mag` | Fujita/Enhanced Fujita magnitude (3–5) |
| `slat` / `slon` | Tornado start coordinates |
| `len` | Path length in miles |
| `wid` | Path width in yards |

**Quick load example:**
```python
import pandas as pd
coastal = pd.read_csv('data/coastal_flood_events.csv')
wildfires = pd.read_csv('data/wildfire_events.csv')
hurricanes = pd.read_csv('data/us_hurricane_landfalls.csv')
tornadoes = pd.read_csv('data/tornadoes_f3plus.csv')
```
