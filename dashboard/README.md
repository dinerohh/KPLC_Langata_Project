# KPLC Langata Grid Intelligence Dashboard

Dark electric monitoring dashboard for the Langata Substation outage prediction system.

## Setup

### 1. Folder structure required
```
dashboard/
├── app.py
├── requirements.txt
└── data/
    ├── master_frame.csv              ← from notebook Section 3
    ├── incidences_Langata_s_s.xlsx   ← raw KPLC data
    ├── feeder_lines.csv              ← from DXF parser
    ├── network_nodes.csv             ← from DXF parser
    └── feeder_topology.csv           ← from DXF parser
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```

### 4. Deploy to Streamlit Cloud (free)
1. Push this folder to a GitHub repository
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set main file path to `app.py`
5. Add any secrets if needed

## Pages

| Page | Content |
|---|---|
| 📡 Overview | KPI cards, monthly trend, cause breakdown, today's risk table |
| 🗺️ Network Map | Interactive feeder map with risk-coloured lines and node markers |
| 📅 7-Day Forecast | Risk heatmap calendar + per-feeder day cards |
| ⚡ Feeder Analysis | Per-feeder deep dive — history, cause types, monthly calendar |
| 🌧️ Weather Correlation | Precipitation/wind vs outage scatter, seasonal breakdown |
| 📊 Reliability | SAIFI/SAIDI by feeder, financial loss analysis |
| 📋 Event Log | Searchable/filterable historical incidence table |

## Notes
- The 7-day forecast uses historical seasonal patterns as a proxy.
  In production, connect to the trained models in `models/saved/`.
- The network map requires `carto-darkmatter` mapbox style (no token needed).
- All data files should be copied from the notebook's `outputs/dashboard/` folder on Drive.
