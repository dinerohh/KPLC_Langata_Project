"""
KPLC Langata Substation — Grid Intelligence Dashboard
Dark electric theme · Light mode toggle · Fast map · Collapsible sidebar
Run:  streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

def hex_to_rgba(hex_color, alpha=1.0):
    """Convert #rrggbb hex colour to rgba() string for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KPLC Langata Grid Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state ─────────────────────────────────────────────────────────────
if "page"       not in st.session_state: st.session_state.page       = "📡 Overview"
if "dark_mode"  not in st.session_state: st.session_state.dark_mode  = True
if "show_nav"   not in st.session_state: st.session_state.show_nav   = True

dark = st.session_state.dark_mode

# ── colour tokens — dark and light ────────────────────────────────────────────
if dark:
    C = {
        "bg":           "#050d1a",
        "panel":        "#091628",
        "panel2":       "#0b1e35",
        "border":       "#0d2847",
        "accent":       "#00c8ff",
        "teal":         "#00ffcc",
        "amber":        "#f0a500",
        "red":          "#ff3860",
        "green":        "#00e676",
        "purple":       "#c77dff",
        "text":         "#e0f4ff",
        "text2":        "#5a8aaa",
        "grid":         "#0d2847",
        "topbar":       "#06111e",
        "sidebar":      "#06111e",
        "mapstyle":     "carto-darkmatter",
        "chart_bg":     "rgba(0,0,0,0)",
        "chart_plot":   "rgba(0,0,0,0)",
    }
else:
    C = {
        "bg":           "#f4f6f9",
        "panel":        "#ffffff",
        "panel2":       "#f0f4f8",
        "border":       "#dde3ec",
        "accent":       "#1a5fa8",
        "teal":         "#00796b",
        "amber":        "#e65100",
        "red":          "#c62828",
        "green":        "#2e7d32",
        "purple":       "#6a1b9a",
        "text":         "#1a2332",
        "text2":        "#4a5568",
        "grid":         "#eef0f4",
        "topbar":       "#ffffff",
        "sidebar":      "#f8fafc",
        "mapstyle":     "carto-positron",
        "chart_bg":     "rgba(0,0,0,0)",
        "chart_plot":   "#ffffff",
    }

FEEDER_COLORS = {
    "SOWETO EX LANGATA":         "#00c8ff",
    "MAGADI  EX LANGATA":        "#f0a500",
    "HARDY EX LANGATA":          "#00ffcc",
    "NGEI EX LANGATA":           "#c77dff",
    "NDALATI EX LANGATA":        "#00e676",
    "KUWINDA EX LANGATA":        "#ff6b6b",
    "KAREN HOSPITAL EX LANGATA": "#ffd166",
    "OTIENDE EX LANGATA":        "#a8dadc",
}
if not dark:
    FEEDER_COLORS = {
        "SOWETO EX LANGATA":         "#1565c0",
        "MAGADI  EX LANGATA":        "#e65100",
        "HARDY EX LANGATA":          "#00695c",
        "NGEI EX LANGATA":           "#6a1b9a",
        "NDALATI EX LANGATA":        "#1b5e20",
        "KUWINDA EX LANGATA":        "#b71c1c",
        "KAREN HOSPITAL EX LANGATA": "#f9a825",
        "OTIENDE EX LANGATA":        "#0277bd",
    }

RISK_COLORS = {
    "LOW":      C["green"],
    "MODERATE": C["amber"],
    "HIGH":     C["red"],
    "CRITICAL": "#ff0055" if dark else "#7f0000",
}

# ── global CSS ─────────────────────────────────────────────────────────────────
shadow = "0 1px 4px rgba(0,0,0,0.06)" if not dark else "0 2px 12px rgba(0,200,255,0.06)"
glow   = f"0 0 12px {C['accent']}33" if dark else "none"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family:'Exo 2',sans-serif;
    background-color:{C['bg']};
    color:{C['text']};
}}
.stApp {{ background:{C['bg']}; }}
#MainMenu, footer, header {{ visibility:hidden; }}
.block-container {{ padding:0 1.2rem 2rem 1.2rem; max-width:100%; }}

/* sidebar */
[data-testid="stSidebar"] {{
    background:{C['sidebar']};
    border-right:1px solid {C['border']};
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {{
    color:{C['text2']} !important;
    font-size:0.75rem;
    letter-spacing:0.06em;
    text-transform:uppercase;
}}

/* top bar */
.topbar {{
    background:{C['topbar']};
    border-bottom:1px solid {C['border']};
    padding:0.55rem 1.2rem;
    display:flex; align-items:center; justify-content:space-between;
    margin:0 -1.2rem 1rem -1.2rem;
    position:sticky; top:0; z-index:999;
    box-shadow:{shadow};
}}
.topbar-brand {{
    font-family:'Rajdhani',sans-serif;
    font-size:1.1rem; font-weight:700;
    color:{C['text']}; letter-spacing:0.04em;
}}
.topbar-sub {{
    font-family:'Share Tech Mono',monospace;
    font-size:0.6rem; color:{C['text2']};
    letter-spacing:0.12em; margin-top:2px;
    display:flex; align-items:center; gap:0.5rem;
}}
.live-dot {{
    width:7px; height:7px; border-radius:50%;
    background:{C['green']};
    animation:pulse-live 2s infinite;
    display:inline-block;
}}
@keyframes pulse-live {{
    0%   {{ box-shadow:0 0 0 0 {C['green']}66; }}
    70%  {{ box-shadow:0 0 0 6px {C['green']}00; }}
    100% {{ box-shadow:0 0 0 0 {C['green']}00; }}
}}
.topbar-time {{
    font-family:'Rajdhani',sans-serif;
    font-size:1.5rem; font-weight:700;
    color:{C['accent']};
    {"text-shadow:0 0 20px " + C['accent'] + ";" if dark else ""}
}}
.topbar-date {{
    font-family:'Share Tech Mono',monospace;
    font-size:0.6rem; color:{C['text2']};
    letter-spacing:0.1em; text-align:right;
}}

/* metric cards */
.metric-card {{
    background:{C['panel']};
    border:1px solid {C['border']};
    border-radius:4px;
    padding:0.85rem 1rem;
    position:relative; overflow:hidden;
    box-shadow:{shadow};
}}
.metric-card::before {{
    content:''; position:absolute;
    top:0; left:0; width:3px; height:100%;
    background:{C['accent']};
    {"box-shadow:0 0 12px " + C['accent'] + ";" if dark else ""}
}}
.metric-card.amber::before {{ background:{C['amber']}; {"box-shadow:0 0 12px "+C['amber']+";" if dark else ""} }}
.metric-card.red::before   {{ background:{C['red']};   {"box-shadow:0 0 12px "+C['red']+";" if dark else ""} }}
.metric-card.teal::before  {{ background:{C['teal']};  {"box-shadow:0 0 12px "+C['teal']+";" if dark else ""} }}
.metric-card.green::before {{ background:{C['green']}; {"box-shadow:0 0 12px "+C['green']+";" if dark else ""} }}
.metric-label {{
    font-family:'Share Tech Mono',monospace;
    font-size:0.6rem; letter-spacing:0.12em;
    text-transform:uppercase; color:{C['text2']};
    margin-bottom:4px;
}}
.metric-value {{
    font-family:'Rajdhani',sans-serif;
    font-size:1.85rem; font-weight:700;
    color:{C['text']}; line-height:1;
}}
.metric-sub {{
    font-family:'Share Tech Mono',monospace;
    font-size:0.62rem; color:{C['text2']}; margin-top:3px;
}}

/* section header */
.sec-hdr {{
    font-family:'Rajdhani',sans-serif;
    font-size:0.72rem; font-weight:600;
    letter-spacing:0.18em; text-transform:uppercase;
    color:{C['accent']};
    border-bottom:1px solid {C['border']};
    padding-bottom:0.35rem; margin-bottom:0.8rem;
    display:flex; align-items:center; gap:0.5rem;
}}
.sec-hdr::before {{
    content:''; width:6px; height:6px; border-radius:50%;
    background:{C['accent']}; flex-shrink:0;
    {"box-shadow:0 0 8px "+C['accent']+";" if dark else ""}
}}

/* risk badges */
.badge {{
    display:inline-block; padding:0.15rem 0.55rem;
    border-radius:2px;
    font-family:'Share Tech Mono',monospace;
    font-size:0.65rem; font-weight:700; letter-spacing:0.08em;
}}
.b-low      {{ background:{"rgba(0,230,118,0.15)" if dark else "#e8f5e9"};
               color:{C['green']};
               border:1px solid {"rgba(0,230,118,0.3)" if dark else "#a5d6a7"}; }}
.b-moderate {{ background:{"rgba(240,165,0,0.15)"  if dark else "#fff3e0"};
               color:{C['amber']};
               border:1px solid {"rgba(240,165,0,0.3)"  if dark else "#ffcc80"}; }}
.b-high     {{ background:{"rgba(255,56,96,0.15)"  if dark else "#ffebee"};
               color:{C['red']};
               border:1px solid {"rgba(255,56,96,0.3)"  if dark else "#ef9a9a"}; }}
.b-critical {{ background:{"rgba(255,0,85,0.25)"   if dark else "#b71c1c"};
               color:{"#ff0055" if dark else "#fff"};
               border:1px solid {"rgba(255,0,85,0.5)"   if dark else "#7f0000"};
               animation:pulse-badge 1s infinite; }}
@keyframes pulse-badge {{
    0%,100% {{ box-shadow:0 0 0 0 rgba(255,0,85,0.4); }}
    50%      {{ box-shadow:0 0 0 4px rgba(255,0,85,0); }}
}}

/* panel */
.panel {{
    background:{C['panel']};
    border:1px solid {C['border']};
    border-radius:4px; padding:0.9rem 1rem;
    box-shadow:{shadow};
}}

/* table */
.tbl {{ width:100%; border-collapse:collapse; font-size:0.77rem; }}
.tbl th {{
    background:{C['panel2']}; color:{C['text2']};
    font-family:'Share Tech Mono',monospace;
    font-size:0.6rem; letter-spacing:0.1em;
    text-transform:uppercase; padding:0.45rem 0.7rem;
    text-align:left; border-bottom:1px solid {C['border']};
    position:sticky; top:0;
}}
.tbl td {{ padding:0.38rem 0.7rem; border-bottom:1px solid {C['border']}33; }}
.tbl tr:hover td {{ background:{"rgba(0,200,255,0.03)" if dark else C['panel2']}; }}

/* heatmap */
.hmap {{ display:grid; gap:3px; }}
.hmap-cell {{
    border-radius:3px; height:30px;
    display:flex; align-items:center; justify-content:center;
    font-family:'Share Tech Mono',monospace;
    font-size:0.65rem; font-weight:700;
    transition:transform 0.15s;
}}
.hmap-cell:hover {{ transform:scale(1.06); }}
.hmap-lbl {{
    font-family:'Share Tech Mono',monospace;
    font-size:0.62rem; color:{C['text2']};
    display:flex; align-items:center;
}}
.hmap-day {{
    font-family:'Share Tech Mono',monospace;
    font-size:0.6rem; font-weight:600; color:{C['text2']};
    text-align:center;
}}

/* day cards */
.day-cards {{ display:grid; grid-template-columns:repeat(7,1fr); gap:0.5rem; margin-top:0.8rem; }}
.day-card {{
    background:{C['panel']};
    border:1px solid {C['border']};
    border-radius:4px; padding:0.65rem 0.4rem; text-align:center;
}}

/* scrollbar */
::-webkit-scrollbar {{ width:4px; height:4px; }}
::-webkit-scrollbar-track {{ background:{C['bg']}; }}
::-webkit-scrollbar-thumb {{ background:{C['border']}; border-radius:2px; }}

/* sidebar toggle button */
[data-testid="collapsedControl"] button {{ color:{C['text']} !important; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
BASE = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_master():
    for name in ["master_frame.csv", "feeder_day_frame.csv"]:
        p = BASE / name
        if p.exists():
            return pd.read_csv(p, parse_dates=["date"])
    st.error("master_frame.csv not found in dashboard/data/")
    st.stop()

@st.cache_data(show_spinner=False)
def load_topology():
    try:
        return (pd.read_csv(BASE / "feeder_lines.csv"),
                pd.read_csv(BASE / "network_nodes.csv"),
                pd.read_csv(BASE / "feeder_topology.csv"))
    except FileNotFoundError:
        return None, None, None

@st.cache_data(show_spinner=False, ttl=300)
def build_forecast(seed):
    df = load_master()
    feeders = sorted(df.feeder.unique())
    today   = pd.Timestamp.today().normalize()
    dates   = [today + timedelta(days=i) for i in range(7)]
    FAULT   = {0:"No Outage", 1:"Loss of Supply",
                2:"Controlled Interruption", 3:"Physical Fault"}
    CREW    = {0:"—", 1:"System Engineers",
                2:"Switching Crew", 3:"Line Maintenance + Tree Cutting"}
    rows = []
    np.random.seed(42)
    for feeder in feeders:
        fdf = df[df.feeder == feeder]
        for i, d in enumerate(dates):
            sm    = fdf[fdf.month == d.month] if "month" in fdf.columns else fdf
            mr    = (sm.outage_class > 0).mean() if len(sm) > 0 else (fdf.outage_class > 0).mean()
            r30   = (fdf.tail(30).outage_class > 0).mean()
            prob  = float(np.clip(0.55*mr + 0.35*r30 + 0.10*np.random.uniform(0, 0.12), 0.01, 0.92))
            risk  = ("CRITICAL" if prob >= 0.55 else "HIGH" if prob >= 0.30
                     else "MODERATE" if prob >= 0.15 else "LOW")
            od    = fdf[fdf.outage_class > 0]
            fc    = int(od.outage_class.mode()[0]) if len(od) > 0 and prob >= 0.30 else 0
            rows.append(dict(feeder=feeder, date=d, prob=round(prob,4), risk=risk,
                             fault_class=fc, fault_label=FAULT[fc], crew=CREW[fc]))
    return pd.DataFrame(rows)

df             = load_master()
lines, nodes, topo = load_topology()
forecast       = build_forecast(len(df))

# Aggregates
rel = df.groupby("feeder").agg(
    saifi_num=("saifi_numerator","sum"),
    saidi_num=("saidi_numerator","sum"),
    customers=("affected_customers","max"),
    events   =("n_outages","sum"),
    loss_mksh=("total_loss_mksh","sum"),
    total_hrs=("total_duration_hrs","sum"),
).reset_index()
rel["SAIFI"] = (rel.saifi_num / rel.customers).round(1)
rel["SAIDI"] = (rel.saidi_num / rel.customers).round(1)
rel["short"] = rel.feeder.str.replace(" EX LANGATA","").str.strip()

events_df = df[df.outage_class > 0].copy()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def short(name): return name.replace(" EX LANGATA","").replace("  "," ").strip()

def risk_color(p):
    if p >= 0.55: return "#ff0055" if dark else "#7f0000"
    if p >= 0.30: return C["red"]
    if p >= 0.15: return C["amber"]
    return C["green"]

def risk_bg(p):
    if p >= 0.55: return ("rgba(255,0,85,0.22)"   if dark else "#ffcdd2")
    if p >= 0.30: return ("rgba(255,56,96,0.15)"  if dark else "#ffebee")
    if p >= 0.15: return ("rgba(240,165,0,0.15)"  if dark else "#fff3e0")
    return ("rgba(0,230,118,0.12)" if dark else "#e8f5e9")

def badge(risk):
    cls = {"LOW":"b-low","MODERATE":"b-moderate","HIGH":"b-high","CRITICAL":"b-critical"}
    c = cls.get(risk, "b-low")
    return '<span class="badge ' + c + '">' + risk + '</span>'

def chart_layout(h=300):
    return dict(
        paper_bgcolor=C["chart_bg"],
        plot_bgcolor =C["chart_plot"],
        font=dict(family="Exo 2,sans-serif", size=11, color=C["text2"]),
        margin=dict(l=8, r=8, t=28, b=8),
        height=h,
        xaxis=dict(gridcolor=C["grid"], zeroline=False,
                   linecolor=C["border"], tickfont=dict(size=10)),
        yaxis=dict(gridcolor=C["grid"], zeroline=False,
                   linecolor=C["border"], tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)" if dark else "white",
                    bordercolor=C["border"], font=dict(size=10),
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — collapsible navigation
# ══════════════════════════════════════════════════════════════════════════════
PAGES = ["📡 Overview","🗺️ Network Map","📅 7-Day Forecast",
         "⚡ Feeder Analysis","🌧️ Weather","📊 Reliability","📋 Event Log"]

with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style='text-align:center;padding:0.6rem 0 1rem 0;
                border-bottom:1px solid {C["border"]};margin-bottom:1rem;'>
        <div style='font-family:"Rajdhani",sans-serif;font-size:1.1rem;
                    font-weight:700;color:{C["accent"]};
                    {"text-shadow:0 0 18px "+C["accent"]+";" if dark else ""}'>
            ⚡ LANGATA
        </div>
        <div style='font-family:"Share Tech Mono",monospace;font-size:0.58rem;
                    color:{C["text2"]};letter-spacing:0.14em;'>
            GRID INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dark / Light toggle
    mode_label = "☀️ Light Mode" if dark else "🌙 Dark Mode"
    if st.button(mode_label, use_container_width=True, key="mode_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:Share Tech Mono,monospace;"
                f"font-size:0.6rem;color:{C['text2']};letter-spacing:0.12em;"
                f"text-transform:uppercase;margin-bottom:0.5rem;'>Navigation</div>",
                unsafe_allow_html=True)

    for pg in PAGES:
        is_active = st.session_state.page == pg
        label = ("▶ " if is_active else "  ") + pg
        if st.button(label, key=f"nav_{pg}", use_container_width=True):
            st.session_state.page = pg
            st.rerun()

    # Active styling
    st.markdown(f"""
    <style>
    [data-testid="stSidebar"] button {{
        background: transparent;
        border: none;
        color: {C['text2']};
        font-size: 0.78rem;
        text-align: left;
        border-radius: 4px;
        margin-bottom: 2px;
        padding: 0.35rem 0.6rem;
    }}
    [data-testid="stSidebar"] button:hover {{
        background: {"rgba(0,200,255,0.08)" if dark else C['panel2']};
        color: {C['text']};
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:Share Tech Mono,monospace;"
                f"font-size:0.6rem;color:{C['text2']};letter-spacing:0.12em;"
                f"text-transform:uppercase;margin-bottom:0.5rem;'>Today's Risk</div>",
                unsafe_allow_html=True)

    today_fc = forecast[forecast.date == forecast.date.min()]
    for _, row in today_fc.sort_values("prob", ascending=False).iterrows():
        color = risk_color(row.prob)
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;padding:0.22rem 0;"
            f"border-bottom:1px solid {C['border']}44;'>"
            f"<span style='font-family:Share Tech Mono,monospace;"
            f"font-size:0.63rem;color:{C['text']};'>{short(row.feeder)}</span>"
            f"<span style='font-family:Share Tech Mono,monospace;"
            f"font-size:0.63rem;font-weight:700;color:{color};'>"
            f"{row.prob*100:.0f}%</span></div>",
            unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-top:1.2rem;padding:0.7rem;
                background:{C['panel2']};
                border:1px solid {C['border']};
                border-radius:4px;'>
        <div style='font-family:Share Tech Mono,monospace;font-size:0.58rem;
                    color:{C['text2']};letter-spacing:0.1em;margin-bottom:0.4rem;'>
            MODEL
        </div>
        <div style='font-size:0.68rem;color:{C['text']};line-height:1.7;'>
            SARIMA + Prophet<br>XGBoost Stage 1<br>XGBoost Stage 2
            <br><span style='color:{C['text2']};'>Threshold: 0.30</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
now = datetime.now()
st.markdown(f"""
<div class="topbar">
  <div>
    <div class="topbar-brand">⚡ KPLC LANGATA SUBSTATION — GRID MONITOR</div>
    <div class="topbar-sub">
      <span class="live-dot"></span>
      66KV / 11KV DISTRIBUTION NETWORK &nbsp;|&nbsp;
      LANGATA, NAIROBI &nbsp;|&nbsp;
      8 ACTIVE FEEDERS &nbsp;|&nbsp;
      HYBRID ML PIPELINE v1.0
    </div>
  </div>
  <div>
    <div class="topbar-time">{now.strftime('%H:%M')}</div>
    <div class="topbar-date">{now.strftime('%A, %d %B %Y')}</div>
  </div>
</div>
""", unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📡 Overview":

    high_risk = int((today_fc.risk.isin(["HIGH","CRITICAL"])).sum())

    # KPI row
    kpis = [
        ("Total Events",    "940",         "Jul 2022 – Apr 2026", ""),
        ("Financial Loss",  "KES 85.9M",   "Cumulative",          "amber"),
        ("Outage Hours",    "2,448",        "All feeders",         "red"),
        ("Energy Loss",     "6.135 GWh",   "Grid-wide",           "teal"),
        ("High-Risk Today", f"{high_risk}/8","Feeders flagged",   "red" if high_risk > 2 else "green"),
    ]
    cols = st.columns(5)
    for col, (label, val, sub, cls) in zip(cols, kpis):
        col.markdown(
            f'<div class="metric-card {cls}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

    # Charts
    c1, c2, c3 = st.columns([2.2, 1.1, 1.1])

    with c1:
        st.markdown(f'<div class="panel"><div class="sec-hdr">Monthly Outage Rate vs Precipitation</div>',
                    unsafe_allow_html=True)
        monthly = (df.groupby(["year","month"])
                   .agg(od=("outage_class", lambda x:(x>0).sum()),
                        tot=("outage_class","count"),
                        pr=("precipitation_sum","mean"))
                   .reset_index())
        monthly["rate"] = (monthly.od/monthly.tot*100).round(1)
        monthly["period"] = monthly.apply(lambda r: f"{int(r.year)}-{int(r.month):02d}", axis=1)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly.period, y=monthly.pr,
                             marker_color="rgba(0,255,204,0.27)", marker_line_width=0,
                             name="Avg Precip (mm)", yaxis="y2"))
        fig.add_trace(go.Scatter(x=monthly.period, y=monthly.rate,
                                 mode="lines+markers",
                                 line=dict(color=C["accent"], width=2.5),
                                 marker=dict(size=4, color=C["accent"]),
                                 fill="tozeroy",
                                 fillcolor="rgba(0,200,255,0.07)",
                                 name="Outage Rate %"))
        lay = chart_layout(260)
        lay["xaxis"].update(tickangle=45, nticks=18, tickfont=dict(size=8))
        lay.update(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                               tickfont=dict(size=9), color=C["text2"]),
                   showlegend=True)
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="panel"><div class="sec-hdr">Fault Distribution</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["Loss of Supply","Controlled","Physical Fault"],
            values=[497, 295, 148], hole=0.55,
            marker=dict(colors=[C["accent"], C["amber"], C["red"]],
                        line=dict(color=C["panel"], width=2)),
            textfont=dict(size=10, color=C["text"]),
        ))
        lay2 = chart_layout(260)
        lay2.update(showlegend=True,
                    legend=dict(orientation="v", x=0.6, y=0.4,
                                font=dict(size=9),
                                bgcolor="rgba(0,0,0,0)"))
        fig2.update_layout(**lay2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="panel"><div class="sec-hdr">Events by Feeder</div>',
                    unsafe_allow_html=True)
        fev = (df.groupby("feeder").n_outages.sum().reset_index()
               .sort_values("n_outages", ascending=True))
        fev["short"] = fev.feeder.apply(short)
        fev["color"] = fev.feeder.map(FEEDER_COLORS)
        fig3 = go.Figure(go.Bar(
            x=fev.n_outages, y=fev.short, orientation="h",
            marker=dict(color=fev.color, opacity=0.85),
            text=fev.n_outages, textposition="outside",
            textfont=dict(size=9, color=C["text2"]),
        ))
        lay3 = chart_layout(260)
        lay3["xaxis"].update(showticklabels=False)
        lay3["yaxis"].update(tickfont=dict(size=9))
        lay3.update(showlegend=False, bargap=0.28)
        fig3.update_layout(**lay3)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Today's risk table
    st.markdown(f'<div class="panel" style="margin-top:0.8rem;">'
                f'<div class="sec-hdr">Today\'s Risk Summary</div>',
                unsafe_allow_html=True)

    hcols = st.columns([1.8,0.8,0.8,1.5,2.2])
    for col, lbl in zip(hcols, ["FEEDER","RISK","PROBABILITY","FAULT TYPE","CREW DISPATCH"]):
        col.markdown(f"<span style='font-family:Share Tech Mono,monospace;"
                     f"font-size:0.6rem;color:{C['text2']};letter-spacing:0.1em;"
                     f"text-transform:uppercase;'>{lbl}</span>",
                     unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{C['border']};margin:0.3rem 0 0.5rem 0;'>",
                unsafe_allow_html=True)

    for _, row in today_fc.sort_values("prob", ascending=False).iterrows():
        rcols = st.columns([1.8,0.8,0.8,1.5,2.2])
        fc_color = FEEDER_COLORS.get(row.feeder, C["accent"])
        rcols[0].markdown(
            f"<span style='font-family:Share Tech Mono,monospace;"
            f"font-size:0.72rem;color:{fc_color};'>⬡ {short(row.feeder)}</span>",
            unsafe_allow_html=True)
        rcols[1].markdown(f"<div style='padding:0.25rem 0;'>{badge(row.risk)}</div>",
                          unsafe_allow_html=True)
        rcols[2].markdown(
            f"<span style='font-family:Rajdhani,sans-serif;font-size:1.05rem;"
            f"font-weight:700;color:{risk_color(row.prob)};'>{row.prob*100:.1f}%</span>",
            unsafe_allow_html=True)
        rcols[3].markdown(
            f"<span style='font-size:0.73rem;color:{C['text']};'>{row.fault_label}</span>",
            unsafe_allow_html=True)
        rcols[4].markdown(
            f"<span style='font-size:0.7rem;color:{C['text2']};'>{row.crew}</span>",
            unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NETWORK MAP  — fast rendering (1 trace per feeder)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Network Map":

    st.markdown(f'<div class="panel"><div class="sec-hdr">Langata Feeder Network — Live Risk View</div>',
                unsafe_allow_html=True)

    if lines is None:
        st.warning("feeder_lines.csv not found in dashboard/data/")
    else:
        tr = forecast[forecast.date == forecast.date.min()].set_index("feeder")
        fig_map = go.Figure()

        # One trace per feeder with None-separated segments (fast)
        for feeder in sorted(lines.feeder_name.unique()):
            fl    = lines[lines.feeder_name == feeder]
            prob  = float(tr.loc[feeder,"prob"]) if feeder in tr.index else 0.05
            risk  = tr.loc[feeder,"risk"]         if feeder in tr.index else "LOW"
            color = risk_color(prob)
            lats, lons = [], []
            for _, seg in fl.iterrows():
                lats += [seg.lat_start, seg.lat_end, None]
                lons += [seg.lon_start, seg.lon_end, None]
            fig_map.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode="lines",
                line=dict(width=2.5, color=color),
                name=f"{short(feeder)} — {risk} ({prob*100:.0f}%)",
                hoverinfo="name",
            ))

        # Secondary substations — one trace per feeder
        for feeder in sorted(nodes.feeder_name.unique()):
            fn    = nodes[(nodes.node_type=="secondary_substation") &
                          (nodes.feeder_name==feeder)]
            prob  = float(tr.loc[feeder,"prob"]) if feeder in tr.index else 0.05
            color = risk_color(prob)
            fig_map.add_trace(go.Scattermapbox(
                lat=fn.lat, lon=fn.lon, mode="markers",
                marker=dict(size=5, color=color, opacity=0.75),
                name=short(feeder), showlegend=False,
                hovertemplate=f"<b>{short(feeder)}</b><br>Risk: {prob*100:.0f}%<extra></extra>",
            ))

        # Switch isolators
        sw = nodes[nodes.node_type=="switch_isolator"]
        if len(sw) > 0:
            fig_map.add_trace(go.Scattermapbox(
                lat=sw.lat, lon=sw.lon, mode="markers",
                marker=dict(size=6, color=C["amber"], symbol="square", opacity=0.8),
                name="Switch / Isolator",
                hovertemplate="Switch Isolator<extra></extra>",
            ))

        # Primary substation
        ps = nodes[nodes.node_type=="primary_substation"].iloc[0]
        fig_map.add_trace(go.Scattermapbox(
            lat=[ps.lat], lon=[ps.lon], mode="markers+text",
            marker=dict(size=14, color=C["accent"]),
            text=["Langata 66kV/11kV"],
            textposition="top right",
            textfont=dict(size=11, color=C["accent"],
                          family="Share Tech Mono,monospace"),
            name="Primary Substation",
            hovertemplate="<b>Langata Primary Substation</b><br>66kV / 11kV<extra></extra>",
        ))

        fig_map.update_layout(
            mapbox=dict(style=C["mapstyle"],
                        center=dict(lat=-1.339, lon=36.757), zoom=12.5),
            paper_bgcolor=C["panel"],
            height=560, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                bgcolor=C["panel"],
                bordercolor=C["border"],
                font=dict(size=10, color=C["text"]),
                x=0.01, y=0.99,
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar":False})

        lcols = st.columns(4)
        for col, (lbl, thr) in zip(lcols, [
            ("LOW < 15%",      0.05),
            ("MODERATE 15–30%",0.20),
            ("HIGH 30–55%",    0.40),
            ("CRITICAL > 55%", 0.70),
        ]):
            clr = risk_color(thr)
            col.markdown(
                f"<div style='display:flex;align-items:center;gap:0.4rem;"
                f"font-size:0.68rem;color:{C['text2']};'>"
                f"<div style='width:14px;height:3px;border-radius:2px;"
                f"background:{clr};'></div>{lbl}</div>",
                unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 7-DAY FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅 7-Day Forecast":

    st.markdown(f'<div class="panel"><div class="sec-hdr">7-Day Outage Risk — All Feeders</div>',
                unsafe_allow_html=True)

    feeders  = sorted(forecast.feeder.unique())
    dates    = sorted(forecast.date.unique())
    day_lbls = [pd.Timestamp(d).strftime("%a\n%d %b") for d in dates]
    n_days   = len(dates)

    hmap_html = f'<div class="hmap" style="grid-template-columns:150px repeat({n_days},1fr);overflow-x:auto;">'
    hmap_html += "<div></div>"
    for dl in day_lbls:
        hmap_html += f"<div class='hmap-day'>{dl.replace(chr(10),'<br>')}</div>"
    for feeder in feeders:
        hmap_html += f"<div class='hmap-lbl'>● {short(feeder)}</div>"
        for d in dates:
            row = forecast[(forecast.feeder==feeder) & (forecast.date==d)]
            if not len(row): hmap_html += "<div class='hmap-cell'></div>"; continue
            p   = float(row.iloc[0].prob)
            clr = risk_color(p)
            bg  = risk_bg(p)
            hmap_html += f"<div class='hmap-cell' style='background:{bg};color:{clr};'>{p*100:.0f}%</div>"
    hmap_html += "</div>"
    st.markdown(hmap_html, unsafe_allow_html=True)

    lcols = st.columns(4)
    for col, (lbl, p_ex) in zip(lcols, [("LOW <15%",0.05),("MODERATE 15–30%",0.20),
                                         ("HIGH 30–55%",0.40),("CRITICAL >55%",0.70)]):
        col.markdown(
            f"<div style='display:flex;align-items:center;gap:0.4rem;"
            f"font-size:0.68rem;color:{C['text2']};margin-top:0.5rem;'>"
            f"<div style='width:12px;height:12px;border-radius:3px;"
            f"background:{risk_bg(p_ex)};border:1px solid {risk_color(p_ex)}44;'></div>"
            f"{lbl}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="panel"><div class="sec-hdr">Feeder Detail</div>',
                unsafe_allow_html=True)

    sel    = st.selectbox("Select feeder", options=feeders,
                          format_func=short, label_visibility="collapsed")
    sel_fc = forecast[forecast.feeder==sel].sort_values("date")
    clr    = FEEDER_COLORS.get(sel, C["accent"])

    fig_l = go.Figure()
    fig_l.add_hrect(y0=0.30, y1=1.0,
                    fillcolor="rgba(255,56,96,0.05)", line_width=0)
    fig_l.add_hline(y=0.30, line_dash="dash", line_color=C["amber"],
                    line_width=1.5,
                    annotation_text="Dispatch threshold (30%)",
                    annotation_font=dict(color=C["amber"], size=10),
                    annotation_position="top left")
    fig_l.add_trace(go.Scatter(
        x=sel_fc.date.dt.strftime("%a %d %b"), y=sel_fc.prob,
        mode="lines+markers",
        line=dict(color=clr, width=2.5),
        marker=dict(size=8, color=clr, line=dict(color=C["panel"], width=2)),
        fill="tozeroy", fillcolor=hex_to_rgba(clr, 0.09),
    ))
    lay_l = chart_layout(200)
    lay_l["yaxis"].update(title="Probability", tickformat=".0%", range=[0,1])
    lay_l.update(showlegend=False)
    fig_l.update_layout(**lay_l)
    st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar":False})

    cards_html = "<div class='day-cards'>"
    for _, row in sel_fc.iterrows():
        c_p = risk_color(row.prob)
        cards_html += (
            f"<div class='day-card' style='border-top:3px solid {c_p};'>"
            f"<div style='font-family:Share Tech Mono,monospace;font-size:0.6rem;"
            f"color:{C['text2']};'>{row.date.strftime('%A')}</div>"
            f"<div style='font-size:0.68rem;color:{C['text2']};'>{row.date.strftime('%d %b')}</div>"
            f"<div style='font-family:Rajdhani,sans-serif;font-size:1.4rem;"
            f"font-weight:700;color:{c_p};'>{row.prob*100:.0f}%</div>"
            f"<div>{badge(row.risk)}</div>"
            f"<div style='font-size:0.62rem;color:{C['text2']};margin-top:0.3rem;'>"
            f"{row.fault_label}</div></div>"
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEEDER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Feeder Analysis":

    sel   = st.selectbox("Select feeder", options=sorted(df.feeder.unique()),
                         format_func=short)
    fdf   = df[df.feeder==sel].sort_values("date").copy()
    clr   = FEEDER_COLORS.get(sel, C["accent"])

    k1,k2,k3,k4 = st.columns(4)
    for col, lbl, val, sub, cls in [
        (k1,"Total Events",   f"{int(fdf.n_outages.sum()):,}",    "",""),
        (k2,"Outage Hours",   f"{fdf.total_duration_hrs.sum():.0f}","hrs","amber"),
        (k3,"Financial Loss", f"KES {fdf.total_loss_mksh.sum():.1f}M","","red"),
        (k4,"Outage Rate",    f"{(fdf.outage_class>0).mean()*100:.1f}%",
         f"{int((fdf.outage_class>0).sum())} days","teal"),
    ]:
        col.markdown(f'<div class="metric-card {cls}">'
                     f'<div class="metric-label">{lbl}</div>'
                     f'<div class="metric-value">{val}</div>'
                     f'<div class="metric-sub">{sub}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns([3,1])

    with c1:
        st.markdown(f'<div class="panel"><div class="sec-hdr">30-Day Rolling Outage Rate</div>',
                    unsafe_allow_html=True)
        fdf["rolling"] = ((fdf.outage_class>0).astype(int)
                           .rolling(30,min_periods=1).mean()*100)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=fdf.date, y=fdf["rolling"].values, mode="lines",
            line=dict(color=clr, width=2),
            fill="tozeroy", fillcolor=hex_to_rgba(clr, 0.07),
            name="30-day rolling rate"))
        out_pts = fdf[fdf.outage_class>0]
        fig_f.add_trace(go.Scatter(
            x=out_pts.date, y=out_pts["rolling"].values, mode="markers",
            marker=dict(size=4, color=C["red"], opacity=0.5),
            name="Outage event"))
        lay_f = chart_layout(260)
        lay_f["yaxis"].update(title="Rate (%)")
        lay_f.update(showlegend=True)
        fig_f.update_layout(**lay_f)
        st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="panel"><div class="sec-hdr">Cause Types</div>',
                    unsafe_allow_html=True)
        ct = (fdf[fdf.cause_type_primary!=""]
              .groupby("cause_type_primary").n_outages.sum()
              .sort_values(ascending=True).reset_index())
        fig_ct = go.Figure(go.Bar(
            x=ct.n_outages,
            y=ct.cause_type_primary.apply(lambda x: x.title()[:20]),
            orientation="h", marker_color=clr, marker_opacity=0.85,
        ))
        lay_ct = chart_layout(260)
        lay_ct["yaxis"].update(tickfont=dict(size=8))
        lay_ct.update(showlegend=False)
        fig_ct.update_layout(**lay_ct)
        st.plotly_chart(fig_ct, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f'<div class="panel" style="margin-top:0.8rem;">'
                f'<div class="sec-hdr">Monthly Outage Calendar</div>',
                unsafe_allow_html=True)
    heat = (fdf.groupby(["year","month"])
            .agg(od=("outage_class", lambda x:(x>0).sum()))
            .reset_index())
    hp   = heat.pivot(index="year",columns="month",values="od").fillna(0)
    mnms = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_h = go.Figure(go.Heatmap(
        z=hp.values,
        x=[mnms[int(c)-1] for c in hp.columns],
        y=[str(y) for y in hp.index],
        colorscale=[[0, C["panel2"]], [1, clr]],
        text=[[str(int(v)) for v in row] for row in hp.values],
        texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="<b>%{y} %{x}</b>: %{z} outage days<extra></extra>",
        colorbar=dict(thickness=10, tickfont=dict(size=9,color=C["text2"]),
                      title=dict(text="Days",font=dict(size=10,color=C["text2"]))),
    ))
    lay_h = chart_layout(170)
    fig_h.update_layout(**lay_h)
    st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WEATHER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌧️ Weather":

    c1,c2 = st.columns(2)

    with c1:
        st.markdown(f'<div class="panel"><div class="sec-hdr">Precipitation vs Outage Rate</div>',
                    unsafe_allow_html=True)
        mwx = (df.groupby(["year","month"])
               .agg(rate=("outage_class",lambda x:(x>0).mean()*100),
                    precip=("precipitation_sum","mean")).reset_index())
        fig_w = go.Figure(go.Scatter(
            x=mwx.precip, y=mwx.rate, mode="markers",
            marker=dict(size=9, color=mwx.rate,
                        colorscale=[[0,C["green"]],[0.5,C["amber"]],[1,C["red"]]],
                        opacity=0.85, line=dict(color=C["panel"], width=1)),
            hovertemplate="Precip: %{x:.1f}mm<br>Rate: %{y:.1f}%<extra></extra>",
        ))
        lay_w = chart_layout(280)
        lay_w["xaxis"].update(title="Avg Precipitation (mm)")
        lay_w["yaxis"].update(title="Outage Rate (%)")
        lay_w.update(showlegend=False)
        fig_w.update_layout(**lay_w)
        st.plotly_chart(fig_w, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="panel"><div class="sec-hdr">Wind Gust vs Outage Rate</div>',
                    unsafe_allow_html=True)
        mwd = (df.groupby(["year","month"])
               .agg(rate=("outage_class",lambda x:(x>0).mean()*100),
                    wind=("wind_gusts_10m_max","mean")).reset_index())
        fig_w2 = go.Figure(go.Scatter(
            x=mwd.wind, y=mwd.rate, mode="markers",
            marker=dict(size=9, color=mwd.rate,
                        colorscale=[[0,C["green"]],[0.5,C["amber"]],[1,C["red"]]],
                        opacity=0.85, line=dict(color=C["panel"], width=1)),
            hovertemplate="Wind: %{x:.1f}m/s<br>Rate: %{y:.1f}%<extra></extra>",
        ))
        lay_w2 = chart_layout(280)
        lay_w2["xaxis"].update(title="Avg Max Wind Gust (m/s)")
        lay_w2["yaxis"].update(title="Outage Rate (%)")
        lay_w2.update(showlegend=False)
        fig_w2.update_layout(**lay_w2)
        st.plotly_chart(fig_w2, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="panel"><div class="sec-hdr">Outage Rate by Kenya Season</div>',
                unsafe_allow_html=True)
    sd_rows = []
    for col, lbl in [("season_long_rains","Long Rains (Mar–May)"),
                     ("season_short_rains","Short Rains (Oct–Dec)"),
                     ("season_long_dry","Long Dry (Jun–Sep)"),
                     ("season_short_dry","Short Dry (Jan–Feb)")]:
        if col in df.columns:
            sub = df[df[col]==1]
            sd_rows.append({"season":lbl,
                             "rate":round((sub.outage_class>0).mean()*100,1),
                             "precip":round(sub.precipitation_sum.mean(),1)})
    sd = pd.DataFrame(sd_rows).sort_values("rate",ascending=False)
    fig_s = go.Figure()
    fig_s.add_trace(go.Bar(
        x=sd.season, y=sd.rate,
        marker_color=[C["red"],C["amber"],C["accent"],C["teal"]],
        text=[f"{v:.1f}%" for v in sd.rate], textposition="outside",
        textfont=dict(size=11,color=C["text"]), name="Outage Rate %"))
    fig_s.add_trace(go.Scatter(
        x=sd.season, y=sd.precip, mode="lines+markers",
        line=dict(color=C["teal"],width=2,dash="dot"),
        marker=dict(size=8), yaxis="y2", name="Avg Precipitation (mm)"))
    lay_s = chart_layout(260)
    lay_s["yaxis"].update(title="Outage Rate (%)")
    lay_s.update(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                             tickfont=dict(size=10), color=C["text2"]),
                 showlegend=True)
    fig_s.update_layout(**lay_s)
    st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RELIABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Reliability":

    worst = rel.loc[rel.SAIFI.idxmax(),"short"]
    k1,k2,k3,k4 = st.columns(4)
    for col, lbl, val, sub, cls in [
        (k1,"Avg Network SAIFI",f"{rel.SAIFI.mean():.0f}","Interruptions/customer/yr","red"),
        (k2,"Avg Network SAIDI",f"{rel.SAIDI.mean():.0f}","Hours/customer/yr","amber"),
        (k3,"Worst SAIFI Feeder",worst,f"{rel.SAIFI.max():.0f} interruptions","red"),
        (k4,"Total Customers",f"{rel.customers.sum():,}","Across 8 feeders",""),
    ]:
        col.markdown(f'<div class="metric-card {cls}">'
                     f'<div class="metric-label">{lbl}</div>'
                     f'<div class="metric-value">{val}</div>'
                     f'<div class="metric-sub">{sub}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        st.markdown(f'<div class="panel"><div class="sec-hdr">SAIFI by Feeder</div>',
                    unsafe_allow_html=True)
        rs = rel.sort_values("SAIFI",ascending=True)
        fig_sf = go.Figure(go.Bar(
            x=rs.SAIFI, y=rs.short, orientation="h",
            marker=dict(color=rs.SAIFI,
                        colorscale=[[0,C["green"]],[0.5,C["amber"]],[1,C["red"]]],
                        opacity=0.88),
            text=[f"{v:.0f}" for v in rs.SAIFI],
            textposition="outside", textfont=dict(size=10,color=C["text2"])))
        lay_sf = chart_layout(290)
        lay_sf["xaxis"].update(title="Interruptions/customer/year")
        lay_sf["yaxis"].update(tickfont=dict(size=10))
        lay_sf.update(showlegend=False)
        fig_sf.update_layout(**lay_sf)
        st.plotly_chart(fig_sf, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="panel"><div class="sec-hdr">SAIDI by Feeder</div>',
                    unsafe_allow_html=True)
        rs2 = rel.sort_values("SAIDI",ascending=True)
        fig_sd = go.Figure(go.Bar(
            x=rs2.SAIDI, y=rs2.short, orientation="h",
            marker=dict(color=rs2.SAIDI,
                        colorscale=[[0,C["teal"]],[0.5,C["amber"]],[1,C["red"]]],
                        opacity=0.88),
            text=[f"{v:.0f}" for v in rs2.SAIDI],
            textposition="outside", textfont=dict(size=10,color=C["text2"])))
        lay_sd = chart_layout(290)
        lay_sd["xaxis"].update(title="Hours/customer/year")
        lay_sd["yaxis"].update(tickfont=dict(size=10))
        lay_sd.update(showlegend=False)
        fig_sd.update_layout(**lay_sd)
        st.plotly_chart(fig_sd, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="panel"><div class="sec-hdr">Financial Loss by Feeder (KES Million)</div>',
                unsafe_allow_html=True)
    rf = rel.sort_values("loss_mksh",ascending=False)
    fig_fin = go.Figure(go.Bar(
        x=rf.short, y=rf.loss_mksh,
        marker_color=[FEEDER_COLORS.get(f,C["accent"]) for f in rf.feeder],
        marker_opacity=0.85,
        text=[f"KES {v:.1f}M" for v in rf.loss_mksh],
        textposition="outside", textfont=dict(size=10,color=C["text2"])))
    lay_fin = chart_layout(250)
    lay_fin["yaxis"].update(title="KES Million")
    lay_fin.update(showlegend=False)
    fig_fin.update_layout(**lay_fin)
    st.plotly_chart(fig_fin, use_container_width=True, config={"displayModeBar":False})
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVENT LOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Event Log":

    st.markdown(f'<div class="panel"><div class="sec-hdr">Historical Outage Event Log</div>',
                unsafe_allow_html=True)

    ev = events_df.copy().sort_values("date", ascending=False)
    f1,f2,f3 = st.columns(3)
    with f1:
        fsel = st.multiselect("Feeder", options=sorted(ev.feeder.unique()),
                              format_func=short, default=[], placeholder="All feeders")
    with f2:
        csel = st.multiselect("Cause Type",
                              options=sorted(ev.cause_type_primary.dropna().unique()),
                              default=[], placeholder="All causes")
    with f3:
        clsel = st.multiselect("Fault Class", options=[1,2,3],
                               format_func=lambda x:{1:"Loss of Supply",
                                                     2:"Controlled",
                                                     3:"Physical Fault"}[x],
                               default=[])

    if fsel:  ev = ev[ev.feeder.isin(fsel)]
    if csel:  ev = ev[ev.cause_type_primary.isin(csel)]
    if clsel: ev = ev[ev.outage_class.isin(clsel)]

    st.markdown(f"<div style='font-size:0.7rem;color:{C['text2']};margin:0.4rem 0;'>"
                f"Showing {min(len(ev),300):,} of {len(ev):,} events</div>",
                unsafe_allow_html=True)

    CL = {1:"Loss of Supply",2:"Controlled",3:"Physical Fault"}
    rows_html = ""
    for _, row in ev.head(300).iterrows():
        fc   = FEEDER_COLORS.get(row.feeder, C["accent"])
        cls  = CL.get(row.outage_class,"—")
        bcls = ("b-low" if row.outage_class==2 else
                "b-moderate" if row.outage_class==1 else "b-high")
        rows_html += (
            f"<tr>"
            f"<td style='font-family:Share Tech Mono,monospace;font-size:0.7rem;"
            f"color:{C['text2']};'>{pd.Timestamp(row.date).strftime('%Y-%m-%d')}</td>"
            f"<td><span style='font-size:0.73rem;font-weight:500;color:{fc};'>"
            f"{short(row.feeder)}</span></td>"
            f"<td><span class='badge {bcls}'>{cls}</span></td>"
            f"<td style='font-size:0.73rem;'>{row.cause_type_primary}</td>"
            f"<td style='font-family:Share Tech Mono,monospace;font-size:0.7rem;"
            f"color:{C['text2']};'>{row.total_duration_hrs:.1f} hrs</td>"
            f"<td style='font-family:Share Tech Mono,monospace;font-size:0.7rem;"
            f"color:{C['text2']};'>KES {row.total_loss_mksh:.3f}M</td>"
            f"</tr>"
        )

    st.markdown(
        f"<div style='overflow-y:auto;max-height:500px;overflow-x:auto;'>"
        f"<table class='tbl'>"
        f"<thead><tr>"
        f"<th>Date</th><th>Feeder</th><th>Fault Class</th>"
        f"<th>Cause Type</th><th>Duration</th><th>Loss</th>"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<div style='margin-top:2rem;padding-top:0.7rem;"
    f"border-top:1px solid {C['border']};"
    f"display:flex;justify-content:space-between;"
    f"font-family:Share Tech Mono,monospace;font-size:0.6rem;color:{C['text2']};'>"
    f"<span>KPLC LANGATA GRID INTELLIGENCE · JKUAT BSc DATA SCIENCE · "
    f"HYBRID ML: SARIMA + PROPHET + XGBOOST</span>"
    f"<span>DATA: JUL 2022 – APR 2026 · 8 FEEDERS · 940 EVENTS</span>"
    f"</div>",
    unsafe_allow_html=True)
