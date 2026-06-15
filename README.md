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
