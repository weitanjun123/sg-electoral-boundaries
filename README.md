# Singapore Electoral Boundaries Timelapse (GE2006 → GE2025)

An interactive map that steps through every official Singapore electoral boundary map published as map data, one general election after another. Land that moved to a different constituency since the previous election is highlighted. Also shows official vote counts (national + per constituency, by party) and tests whether MRT access or breakdowns are linked to PAP vote swings.

Part of the DUCKTRADER4444 deep-dive series.

## What's here

| Path | What |
|---|---|
| `data/raw/` | ELD "Electoral Boundary" GeoJSON for 2006, 2011, 2015, 2020, 2025 (unchanged copies of the data.gov.sg files) |
| `scripts/process.py` | Builds `docs/data.json`: simplified SVG paths, 100 m grid change detection, lineage, churn heatmap |
| `data/raw/results/` | ELD results by candidate, GE1955–GE2025 (data.gov.sg) |
| `data/raw/hdb/`, `data/raw/mrt/` | HDB blocks (units, year completed, coordinates) and MRT stations, from `sg-transit-planner` |
| `scripts/analysis.py` | Vote totals, notional previous results, MRT access features, regressions → `docs/results.json` + `docs/results.txt` |
| `docs/` | The page (`index.html` + `data.json` + `results.json`), ready for GitHub Pages |

## Rebuild

```bash
python3 scripts/process.py data/raw docs/data.json   # needs numpy, scipy, matplotlib
python3 scripts/analysis.py data/raw docs/results.json
```

## Method

- All years are clipped to the 2025 coastline (older files include territorial sea), so earlier maps also show land reclaimed later.
- **Redrawn**: on a 100 m grid, a spot keeps its identity only if its old constituency mainly became its new one *and* the new one was mainly built from the old one. Renames (e.g. Moulmein-Kallang → Jalan Besar) count as redrawn.
- Shares are by **land area**, not by voters.
- Seat counts per constituency are from Wikipedia's lists of Singaporean electoral divisions. The script asserts that each year's total matches the official count (84 / 87 / 89 / 93 / 97) and the number of GRCs and SMCs.
- Pre-2006 boundaries (1988–2001) aren't published as map data, so they're not included.

## Sources

- Elections Department (ELD) via [data.gov.sg](https://data.gov.sg/datasets?agencies=Elections%20Department%20(ELD))
- Mirrors used to fetch the raw files: [tinsights/salamander](https://github.com/tinsights/salamander) (2006–2015), [davidcagoh/sg-redistricting-2025](https://github.com/davidcagoh/sg-redistricting-2025) (2020, 2025)
