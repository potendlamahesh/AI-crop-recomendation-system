import numbers
import os
import pickle
from datetime import datetime
from html import escape
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")

# Prediction history is kept per visitor in st.session_state.
# prediction_history.csv is NOT used by the app any more (shared files would
# let one visitor see another visitor's records).
HISTORY_KEY = "prediction_history"

st.set_page_config(
    page_title="AI Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="auto",  # expanded on desktop, collapsed on phones
)

# Model accuracy is read from crop_model.pkl metadata.
# No hardcoded fallback is used.

# Brand palette (UI only)
GREEN = "#2E7D32"
GREEN_DARK = "#1B5E20"
GREEN_LIGHT = "#E8F5E9"
GREEN_MID = "#66BB6A"
GREEN_PALE = "#A5D6A7"
TEXT = "#263238"
MUTED = "#546E7A"
LINE = "#E1E9E1"

FONT_STACK = (
    "'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, "
    "'Segoe UI', Roboto, sans-serif"
)

# Navigation: (page name, icon name). The page names are what the
# page-routing code below compares against.
PAGES = [
    ("Dashboard", "home"),
    ("Crop Prediction", "sprout"),
    ("Model Insights", "bar-chart"),
    ("Prediction History", "history"),
]
PAGE_NAMES = [name for name, _ in PAGES]

# Friendly display names for the model's feature columns (display only)
FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}


# ==========================================
# 2. LOAD MODEL AND DATASET
# ==========================================


def file_stamp(path):
    """Size and modified time of a file.

    Used as a cache key so the cache refreshes when the file changes
    (for example after a new model is uploaded).
    """

    info = os.stat(path)

    return (info.st_size, info.st_mtime_ns)


@st.cache_resource
def load_model(path, stamp):
    """Load the saved model dictionary. `stamp` only keys the cache."""

    return joblib.load(path)


@st.cache_data
def load_dataset(path, stamp):
    """Load the training dataset. `stamp` only keys the cache."""

    return pd.read_csv(path)


# ==========================================
# 3. UI KIT (frontend only)
# ==========================================

# Simple line icons (24x24 grid, stroke based)
ICONS = {
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    "sprout": '<path d="M7 20h10"/><path d="M10 20c5.5-2.5.8-6.4 3-10"/><path d="M9.5 9.4c1.1.8 1.8 2.2 2.3 3.7-2 .4-3.5.4-4.8-.3-1.2-.6-2.3-1.9-3-4.2 2.8-.5 4.4 0 5.5.8z"/><path d="M14.1 6a7 7 0 0 0-1.1 4c1.9-.1 3.3-.6 4.3-1.4 1-1 1.6-2.3 1.7-4.6-2.7.1-4 1-4.9 2z"/>',
    "home": '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/><path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "bar-chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "history": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "tree": '<circle cx="12" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="19" r="2"/><path d="M12 7v4"/><path d="M12 11H6v6"/><path d="M12 11h6v6"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "flask": '<path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/>',
    "cloud-sun": '<path d="M12 2v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="M20 12h2"/><path d="m19.07 4.93-1.41 1.41"/><path d="M15.947 12.65a4 4 0 0 0-5.925-4.128"/><path d="M13 22H7a5 5 0 1 1 4.9-6H13a3 3 0 0 1 0 6Z"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "trophy": '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>',
    "clipboard": '<rect width="8" height="4" x="8" y="2" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>',
    "hash": '<path d="M4 9h16"/><path d="M4 15h16"/><path d="M10 3 8 21"/><path d="m16 3-2 18"/>',
    "star": '<path d="M11.525 2.295a.53.53 0 0 1 .95 0l2.31 4.679a2.123 2.123 0 0 0 1.595 1.16l5.166.756a.53.53 0 0 1 .294.904l-3.736 3.638a2.123 2.123 0 0 0-.611 1.878l.882 5.14a.53.53 0 0 1-.771.56l-4.618-2.428a2.122 2.122 0 0 0-1.973 0L6.396 21.01a.53.53 0 0 1-.77-.56l.881-5.139a2.122 2.122 0 0 0-.611-1.879L2.16 9.795a.53.53 0 0 1 .294-.906l5.165-.755a2.122 2.122 0 0 0 1.597-1.16z"/>',
}


def icon(name, size=20):
    """Return an inline SVG icon that inherits the current text colour."""

    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true">'
        f"{ICONS[name]}</svg>"
    )


def render_html(markup):
    """Render an HTML snippet.

    Lines are joined first so Markdown never mistakes indented HTML for a
    code block.
    """

    compact = " ".join(line.strip() for line in markup.strip().splitlines())

    st.markdown(compact, unsafe_allow_html=True)


def stretch_call(widget, *args, **kwargs):
    """Call a Streamlit widget at full width across versions."""

    try:
        return widget(*args, width="stretch", **kwargs)
    except TypeError as error:
        error_message = str(error)
        if "width" in error_message and "unexpected keyword" in error_message:
            return widget(*args, use_container_width=True, **kwargs)
        raise


def card_container(key):
    """A bordered card container (styled in the CSS through its key)."""

    try:
        return st.container(border=True, key=key)
    except TypeError:
        return st.container(border=True)


def nav_icon_css():
    """CSS that draws one icon in front of each sidebar navigation item."""

    rules = []

    for position, (_, icon_name) in enumerate(PAGES, start=1):

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
            'fill="none" stroke="black" stroke-width="1.8" '
            'stroke-linecap="round" stroke-linejoin="round">'
            f"{ICONS[icon_name]}</svg>"
        )

        uri = 'url("data:image/svg+xml,' + quote(svg) + '")'

        rules.append(
            f'[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:nth-child({position}) label::before,'
            f'[data-testid="stSidebar"] [role="radiogroup"] > label:nth-child({position})::before'
            f"{{-webkit-mask-image:{uri};mask-image:{uri};}}"
        )

    return "\n".join(rules)


APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --green: #2E7D32;
    --green-dark: #1B5E20;
    --green-light: #E8F5E9;
    --green-mid: #66BB6A;
    --bg: #F8FAF8;
    --white: #FFFFFF;
    --text: #263238;
    --muted: #546E7A;
    --line: #E1E9E1;
    --shadow: 0 1px 2px rgba(27, 94, 32, .05), 0 10px 28px rgba(27, 94, 32, .07);
    --font: __FONT__;
}

/* ---------- Base ---------- */
.stApp { background: var(--bg); color: var(--text); font-family: var(--font); }
.stApp p, .stApp label, .stApp li, .stApp input, .stApp button,
.stApp textarea, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: var(--font);
}
.stApp p, .stApp li { color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { max-width: 1180px; padding: 2.2rem 2rem 4rem; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--white);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: .25rem; }

.brand { display: flex; align-items: center; gap: .8rem; padding: .2rem .2rem 1.4rem; }
.brand-mark {
    width: 44px; height: 44px; flex: none; border-radius: 13px; color: #fff;
    display: grid; place-items: center;
    background: linear-gradient(140deg, var(--green) 0%, var(--green-dark) 100%);
    box-shadow: 0 6px 14px rgba(27, 94, 32, .28);
}
.brand-name { font-weight: 800; font-size: 1.02rem; line-height: 1.15; color: var(--green-dark); }
.brand-tag { font-size: .78rem; color: var(--muted); margin-top: .15rem; }

.side-note {
    margin-top: 1.4rem; padding: .95rem 1rem; border-radius: 14px;
    background: var(--green-light); color: var(--green-dark);
    font-size: .84rem; line-height: 1.5;
}
.side-foot { margin-top: 1rem; font-size: .75rem; color: var(--muted); line-height: 1.5; }

/* navigation radio -> menu items */
[data-testid="stSidebar"] [data-testid="stRadioGroup"],
[data-testid="stSidebar"] [role="radiogroup"] { gap: .3rem; }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div { width: 100%; }

[data-testid="stSidebar"] label[data-testid="stRadioOption"],
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    display: flex; align-items: center; gap: .75rem; width: 100%;
    margin: 0; padding: .7rem .9rem; border-radius: 12px; cursor: pointer;
    transition: background .15s ease, color .15s ease;
}
/* hide the native radio dot */
[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]),
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child { display: none; }

[data-testid="stSidebar"] label[data-testid="stRadioOption"]::before,
[data-testid="stSidebar"] label[data-baseweb="radio"]::before {
    content: ""; width: 20px; height: 20px; flex: none;
    background-color: currentColor;
    -webkit-mask-size: contain; mask-size: contain;
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    -webkit-mask-position: center; mask-position: center;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"] p,
[data-testid="stSidebar"] label[data-baseweb="radio"] p {
    font-weight: 600; font-size: .95rem; color: inherit; margin: 0;
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"],
[data-testid="stSidebar"] label[data-baseweb="radio"] { color: var(--muted); }
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover,
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    background: var(--green-light); color: var(--green-dark);
}
[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"],
[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked),
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
    background: var(--green); color: #fff;
    box-shadow: 0 6px 14px rgba(46, 125, 50, .28);
}

/* ---------- Page header ---------- */
.page-header { display: flex; align-items: center; gap: 1rem; margin: 0 0 1.6rem; }
.ph-icon {
    width: 52px; height: 52px; flex: none; border-radius: 15px;
    display: grid; place-items: center;
    background: var(--green-light); color: var(--green);
}
.ph-title { font-size: 1.85rem; font-weight: 800; color: var(--green-dark); line-height: 1.15; letter-spacing: -.01em; }
.ph-sub { margin-top: .25rem; color: var(--muted); font-size: 1rem; line-height: 1.5; }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden; border-radius: 22px;
    padding: 2.6rem 2.6rem 2.4rem; color: #fff;
    background: linear-gradient(120deg, var(--green-dark) 0%, #236B28 55%, var(--green) 100%);
    box-shadow: 0 18px 40px rgba(27, 94, 32, .22);
}
.hero-art { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.hero-art path { fill: none; stroke: rgba(255, 255, 255, .10); stroke-width: 1.4; }
.hero-leaf { position: absolute; right: -30px; bottom: -40px; width: 280px; height: 280px; color: rgba(255, 255, 255, .10); pointer-events: none; }
.hero-body { position: relative; max-width: 660px; }
.hero-pill {
    display: inline-block; padding: .35rem .85rem; border-radius: 999px;
    background: rgba(255, 255, 255, .16); font-size: .85rem; font-weight: 600; color: #fff;
}
.hero-title { margin: 1rem 0 .7rem; font-size: 2.6rem; font-weight: 800; line-height: 1.1; letter-spacing: -.02em; color: #fff; }
.hero .hero-lede { margin: 0 0 1.4rem; font-size: 1.05rem; line-height: 1.6; color: rgba(255, 255, 255, .88); max-width: 560px; }
.chip-row { display: flex; flex-wrap: wrap; gap: .45rem; }
.chip {
    padding: .3rem .75rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
    background: rgba(255, 255, 255, .12); border: 1px solid rgba(255, 255, 255, .22); color: #fff;
}

/* ---------- Section headings ---------- */
.section-head { margin: 2.2rem 0 1rem; }
.section-title { font-size: 1.3rem; font-weight: 800; color: var(--green-dark); letter-spacing: -.01em; }
.section-sub { margin-top: .2rem; color: var(--muted); font-size: .95rem; line-height: 1.5; }

/* ---------- Cards ---------- */
.card, .st-key-card_soil, .st-key-card_weather,
[class*="st-key-chart_"], [class*="st-key-panel_"],
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--white);
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    box-shadow: var(--shadow);
}
.st-key-card_soil, .st-key-card_weather,
[class*="st-key-chart_"], [class*="st-key-panel_"] { padding: 1.35rem 1.5rem 1.5rem !important; }
[data-testid="stForm"] { border: 0 !important; padding: 0 !important; background: transparent !important; }

.card-head { display: flex; align-items: center; gap: .85rem; margin-bottom: 1.1rem; }
.ch-icon {
    width: 40px; height: 40px; flex: none; border-radius: 12px;
    display: grid; place-items: center; background: var(--green-light); color: var(--green);
}
.ch-title { font-size: 1.08rem; font-weight: 800; color: var(--green-dark); line-height: 1.2; }
.ch-sub { font-size: .85rem; color: var(--muted); margin-top: .1rem; }

/* ---------- Stat cards ---------- */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); gap: 1rem; }
.stat {
    display: flex; gap: .95rem; align-items: flex-start; padding: 1.15rem 1.25rem;
    background: var(--white); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow);
}
.stat-icon {
    width: 44px; height: 44px; flex: none; border-radius: 13px;
    display: grid; place-items: center; background: var(--green-light); color: var(--green);
}
.stat-body { min-width: 0; }
.stat-label { font-size: .85rem; font-weight: 600; color: var(--muted); }
.stat-value { margin-top: .2rem; font-size: 1.65rem; font-weight: 800; color: var(--green-dark); line-height: 1.2; letter-spacing: -.01em; overflow-wrap: anywhere; }
.stat-value.is-text { font-size: 1.3rem; }
.stat-note { margin-top: .25rem; font-size: .78rem; color: var(--muted); line-height: 1.4; }

/* ---------- How it works / features ---------- */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }
.step { padding: 1.25rem 1.3rem; background: var(--white); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); }
.step-num {
    width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center;
    background: var(--green); color: #fff; font-weight: 800; font-size: .95rem; margin-bottom: .85rem;
}
.step-title { font-weight: 800; color: var(--green-dark); font-size: 1.02rem; }
.step-text { margin-top: .35rem; font-size: .9rem; line-height: 1.55; color: var(--muted); }

.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; }
.feature { display: flex; gap: .9rem; padding: 1.15rem 1.25rem; background: var(--green-light); border-radius: 16px; }
.feature-icon { flex: none; color: var(--green); margin-top: .1rem; }
.feature-title { font-weight: 800; color: var(--green-dark); font-size: .98rem; }
.feature-text { margin-top: .25rem; font-size: .88rem; line-height: 1.5; color: #355A3A; }

/* ---------- Callouts ---------- */
.callout { display: flex; gap: .8rem; align-items: flex-start; padding: .95rem 1.1rem; border-radius: 14px; margin: 1rem 0; font-size: .93rem; line-height: 1.55; }
.callout svg { flex: none; margin-top: .1rem; }
.callout.success { background: var(--green-light); color: var(--green-dark); }
.callout.warn { background: #FFF6E0; color: #6B4A00; }
.callout.info { background: #EEF3F5; color: #2F4550; }
.callout ul { margin: .35rem 0 0; padding-left: 1.1rem; }
.callout li { color: inherit; }
.callout-title { font-weight: 700; }

/* ---------- Recommendation cards ---------- */
.rec-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1rem; }
.rec { padding: 1.3rem 1.4rem 1.4rem; background: var(--white); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); }
.rec-top { display: flex; align-items: center; justify-content: space-between; color: var(--muted); font-size: .85rem; font-weight: 600; }
.rec-rank { display: inline-flex; align-items: center; gap: .4rem; }
.rec-badge { width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; background: var(--green-light); color: var(--green-dark); font-weight: 800; font-size: .85rem; }
.rec-crop { margin: .9rem 0 .2rem; font-size: 1.45rem; font-weight: 800; color: var(--green-dark); letter-spacing: -.01em; overflow-wrap: anywhere; }
.rec-score { font-size: 2.1rem; font-weight: 800; color: var(--green); line-height: 1.1; }
.rec-score small { font-size: 1.05rem; font-weight: 700; margin-left: 2px; }
.rec-bar { margin-top: .9rem; height: 8px; border-radius: 999px; background: var(--green-light); overflow: hidden; }
.rec-bar > div { height: 100%; border-radius: 999px; background: var(--green); }
.rec-note { margin-top: .55rem; font-size: .78rem; color: var(--muted); }

.rec.featured {
    background: linear-gradient(145deg, var(--green-dark) 0%, var(--green) 100%);
    border-color: transparent; box-shadow: 0 18px 36px rgba(27, 94, 32, .30);
}
.rec.featured .rec-top, .rec.featured .rec-crop, .rec.featured .rec-score, .rec.featured .rec-note { color: #fff; }
.rec.featured .rec-note { color: rgba(255, 255, 255, .82); }
.rec.featured .rec-badge { background: rgba(255, 255, 255, .2); color: #fff; }
.rec.featured .rec-bar { background: rgba(255, 255, 255, .22); }
.rec.featured .rec-bar > div { background: #fff; }

/* ---------- Native widgets ---------- */
[data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .9rem; color: var(--text); }

[data-testid="stNumberInputContainer"], [data-baseweb="input"], [data-baseweb="base-input"] {
    background: #fff !important; border-radius: 12px !important;
}
[data-testid="stNumberInputContainer"] {
    border: 1px solid var(--line) !important; box-shadow: none;
}
[data-testid="stNumberInputContainer"]:focus-within {
    border-color: var(--green) !important; box-shadow: 0 0 0 3px rgba(46, 125, 50, .16) !important;
}
[data-testid="stNumberInputField"] { color: var(--text); font-weight: 600; background: transparent; }

[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"],
button[kind="primary"], button[kind="primaryFormSubmit"] {
    background: var(--green); color: #fff; border: 0; border-radius: 12px;
    padding: .8rem 1.5rem; font-weight: 700; font-size: 1rem;
    box-shadow: 0 8px 18px rgba(46, 125, 50, .28);
    transition: background .15s ease, transform .15s ease;
}
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p { color: #fff; font-weight: 700; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover,
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover { background: var(--green-dark); color: #fff; border: 0; }
[data-testid="stBaseButton-primaryFormSubmit"] { min-height: 3.2rem; }
[data-testid="stBaseButton-primary"]:focus-visible, [data-testid="stBaseButton-primaryFormSubmit"]:focus-visible,
[data-testid="stBaseButton-secondary"]:focus-visible,
button[kind="primary"]:focus-visible, button[kind="primaryFormSubmit"]:focus-visible {
    outline: none; box-shadow: 0 0 0 3px rgba(46, 125, 50, .35);
}
[data-testid="stBaseButton-primary"]:focus:not(:active), [data-testid="stBaseButton-primaryFormSubmit"]:focus:not(:active) {
    border-color: transparent; color: #fff;
}
[data-testid="stBaseButton-primary"]:active, [data-testid="stBaseButton-primaryFormSubmit"]:active { background: var(--green-dark); color: #fff; }

[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
    background: #fff; color: var(--green-dark); border: 1.5px solid var(--green); border-radius: 12px;
    padding: .65rem 1.25rem; font-weight: 700;
}
[data-testid="stBaseButton-secondary"] p { color: var(--green-dark); font-weight: 700; }
[data-testid="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover { background: var(--green-light); border-color: var(--green-dark); color: var(--green-dark); }

[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stPlotlyChart"] { margin-top: -.25rem; }
.js-plotly-plot .plotly .modebar { display: none !important; }

/* ---------- Top page menu: phones and tablets only ---------- */
.st-key-nav_top { display: none; }

/* ---------- Tablet & mobile ---------- */
@media (max-width: 992px) {
    .st-key-nav_top { display: block; margin: -.25rem 0 1rem; }
    .st-key-nav_top [role="radiogroup"] { flex-wrap: wrap; gap: .45rem; }
    .st-key-nav_top label {
        background: #fff; border: 1.5px solid var(--line); border-radius: 999px;
        padding: .45rem .95rem; min-height: 2.5rem; align-items: center; margin: 0;
    }
    .st-key-nav_top label:has(input:checked) { background: var(--green-light); border-color: var(--green); }
    .block-container { padding: 3.6rem 1.25rem 3rem; }
    .hero { padding: 2rem 1.75rem; }
    .hero-title { font-size: 2.1rem; }
}
@media (max-width: 640px) {
    /* Keep paired inputs (N/P, K/pH, Temp/Humidity) side by side on phones */
    .st-key-card_soil [data-testid="stHorizontalBlock"],
    .st-key-card_weather [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .6rem; }
    .st-key-card_soil [data-testid="stColumn"],
    .st-key-card_weather [data-testid="stColumn"] { min-width: 0 !important; flex: 1 1 0 !important; width: auto !important; }
    /* Compact two-column stat cards. A lone last card spans the full width. */
    .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .65rem; }
    .stat-grid .stat:last-child:nth-child(odd) { grid-column: 1 / -1; }
    .stat { flex-direction: column; gap: .55rem; padding: .85rem .9rem; border-radius: 16px; }
    .stat-icon { width: 36px; height: 36px; border-radius: 11px; }
    .stat-value { font-size: 1.35rem; }
    .stat-value.is-text { font-size: 1.1rem; }
    /* Easier to tap */
    [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"],
    [data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button { min-height: 3rem; width: 100%; }
    body, .block-container { overflow-x: hidden; }
    .block-container { padding: 3.6rem .9rem 3rem; }
    .hero { padding: 1.6rem 1.25rem; border-radius: 18px; }
    .hero-title { font-size: 1.7rem; }
    .hero .hero-lede { font-size: .98rem; }
    .hero-leaf { width: 180px; height: 180px; }
    .page-header { gap: .8rem; }
    .ph-icon { width: 44px; height: 44px; }
    .ph-title { font-size: 1.45rem; }
    .section-title { font-size: 1.15rem; }
    .st-key-card_soil, .st-key-card_weather,
    [class*="st-key-chart_"], [class*="st-key-panel_"] { padding: 1.05rem 1rem 1.15rem !important; }
    .rec-crop { font-size: 1.25rem; }
}
"""


def inject_css():
    """Inject the global stylesheet (fonts, cards, sidebar, widgets)."""

    css = APP_CSS.replace("__FONT__", FONT_STACK) + "\n" + nav_icon_css()

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------- Reusable UI components ----------


def page_header(title, subtitle, icon_name):

    render_html(f"""
        <div class="page-header">
            <div class="ph-icon">{icon(icon_name, 26)}</div>
            <div>
                <div class="ph-title" role="heading" aria-level="1">{escape(title)}</div>
                <div class="ph-sub">{escape(subtitle)}</div>
            </div>
        </div>
        """)


def section_heading(title, subtitle=None):

    sub = f'<div class="section-sub">{escape(subtitle)}</div>' if subtitle else ""

    render_html(f"""
        <div class="section-head">
            <div class="section-title" role="heading" aria-level="2">{escape(title)}</div>
            {sub}
        </div>
        """)


def card_heading(title, subtitle, icon_name):

    render_html(f"""
        <div class="card-head">
            <div class="ch-icon">{icon(icon_name, 22)}</div>
            <div>
                <div class="ch-title">{escape(title)}</div>
                <div class="ch-sub">{escape(subtitle)}</div>
            </div>
        </div>
        """)


def stat_cards(items):
    """items: list of dicts with icon, label, value and an optional note."""

    cards = []

    for item in items:

        note = (
            f'<div class="stat-note">{escape(item["note"])}</div>'
            if item.get("note")
            else ""
        )

        value_class = "stat-value is-text" if item.get("is_text") else "stat-value"

        cards.append(f"""
            <div class="stat">
                <div class="stat-icon">{icon(item["icon"], 22)}</div>
                <div class="stat-body">
                    <div class="stat-label">{escape(item["label"])}</div>
                    <div class="{value_class}">{escape(str(item["value"]))}</div>
                    {note}
                </div>
            </div>
            """)

    render_html(f'<div class="stat-grid">{"".join(cards)}</div>')


def callout(kind, message, icon_name, items=None, title=None):
    """A soft status message. kind: success | warn | info."""

    title_html = f'<div class="callout-title">{escape(title)}</div>' if title else ""

    list_html = (
        "<ul>" + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"
        if items
        else ""
    )

    render_html(f"""
        <div class="callout {kind}">
            {icon(icon_name, 20)}
            <div>{title_html}<div>{escape(message)}</div>{list_html}</div>
        </div>
        """)


def recommendation_cards(crops, scores):
    """Three result cards; the first-ranked crop gets the premium design."""

    labels = ["Best match", "Second match", "Third match"]

    cards = []

    for rank, (crop, score) in enumerate(zip(crops, scores)):

        featured = " featured" if rank == 0 else ""

        width = max(0.0, min(100.0, float(score)))

        cards.append(f"""
            <div class="rec{featured}">
                <div class="rec-top">
                    <span class="rec-rank">{icon("trophy", 16) if rank == 0 else ""}{labels[rank]}</span>
                    <span class="rec-badge">{rank + 1}</span>
                </div>
                <div class="rec-crop">{escape(str(crop).title())}</div>
                <div class="rec-score">{float(score):.2f}<small>%</small></div>
                <div class="rec-bar"><div style="width:{width:.2f}%"></div></div>
                <div class="rec-note">Model-estimated score</div>
            </div>
            """)

    render_html(f'<div class="rec-grid">{"".join(cards)}</div>')


def round_bars(fig, radius):
    """Round the bar ends (older Plotly versions lack this option)."""

    try:
        fig.update_traces(marker_cornerradius=radius)
    except ValueError:
        pass


def style_figure(fig, height=360, left_margin=8, bar_radius=6):
    """One consistent Plotly look for every chart."""

    fig.update_layout(
        height=height,
        margin=dict(l=left_margin, r=24, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, color=TEXT, size=13),
        showlegend=False,
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=LINE, font_size=13),
    )

    # Let Plotly grow the margins so tick labels and axis titles never clip
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)

    try:
        # small gap between tick labels and bars (newer Plotly only)
        fig.update_yaxes(ticklabelstandoff=8)
    except ValueError:
        pass

    round_bars(fig, bar_radius)

    return fig


def show_chart(fig):
    """Render a Plotly figure full width, without the toolbar."""

    stretch_call(st.plotly_chart, fig, theme=None, config={"displayModeBar": False})


def go_to_page(page_name):
    """Button callback that switches both navigation menus."""

    st.session_state["nav"] = page_name
    st.session_state["nav_top"] = page_name


def sync_from_sidebar():
    st.session_state["nav_top"] = st.session_state["nav"]


def sync_from_top():
    st.session_state["nav"] = st.session_state["nav_top"]


def get_model_display_name(saved_data, model):
    """Short, readable model name (display only)."""

    name = saved_data.get("model_name")

    if not name:

        raw = type(model).__name__

        name = "".join(
            f" {ch}" if ch.isupper() and i else ch for i, ch in enumerate(raw)
        )

    name = str(name).strip()

    if name.lower().endswith(" classifier"):
        name = name[: -len(" classifier")]

    return name


def get_model_accuracy(saved_data):
    """Return the saved accuracy as a percentage (0-100), or None.

    Accepts Python and NumPy numbers. A value from 0 to 1 is read as a
    fraction (0.9954 -> 99.54). A value above 1 and up to 100 is read as
    already a percentage. Missing, non-numeric, boolean, non-finite or
    out-of-range values give None. Nothing is invented.
    """

    if not isinstance(saved_data, dict):
        return None

    for key in ("accuracy", "test_accuracy", "model_accuracy"):

        value = saved_data.get(key)

        if isinstance(value, (bool, np.bool_)):
            continue

        if not isinstance(value, numbers.Real):
            continue

        accuracy = float(value)

        if not np.isfinite(accuracy):
            continue

        if 0 <= accuracy <= 1:
            return accuracy * 100

        if 1 < accuracy <= 100:
            return accuracy

    return None


def format_accuracy(accuracy):
    """Text for an accuracy percentage. Never formats None as a number."""

    if accuracy is None:
        return "Not available"

    return f"{accuracy:.2f}%"


def accuracy_stat(saved_data, model_name):
    """The 'Model Accuracy' card used by every page (one source of truth)."""

    accuracy = get_model_accuracy(saved_data)

    if accuracy is None:
        return {
            "icon": "target",
            "label": "Model Accuracy",
            "value": "Not available",
            "note": "Accuracy metadata was not found in the model file",
        }

    kind = saved_data.get("accuracy_type")

    if kind == "test_split_accuracy":
        label = "Test Split Accuracy"
        note = f"{model_name}. Measured on held-out test data, not a real-world guarantee"
    elif kind == "cross_validation_accuracy":
        label = "Cross-Validation Accuracy"
        note = (
            f"{model_name}. 5-fold cross-validation. Some crops use range-based "
            "data, so real-world accuracy is lower"
        )
    else:
        label = "Model Accuracy"
        note = f"{model_name}. Saved with the model, evaluation type not recorded"

    return {
        "icon": "target",
        "label": label,
        "value": format_accuracy(accuracy),
        "note": note,
    }


def build_compact_history(history):
    """Group each saved prediction into six readable columns.

    Display only: it reads the history table and never changes the CSV.
    Returns None when the file lacks any of the expected columns.
    """

    needed = [
        "Date_Time",
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Soil_pH",
        "Temperature",
        "Humidity",
        "Rainfall",
        "Top_1_Crop",
        "Top_1_Score",
        "Top_2_Crop",
        "Top_2_Score",
        "Top_3_Crop",
        "Top_3_Score",
    ]

    if not all(column in history.columns for column in needed):
        return None

    def number(value, spec="g"):

        try:

            if pd.isna(value):
                return "-"

            return format(float(value), spec)

        except (TypeError, ValueError):

            return "-"

    rows = []

    for record in history.to_dict("records"):

        row = {
            "Date & Time": str(record["Date_Time"]),
            "Soil: N · P · K · pH": " · ".join(
                number(record[column])
                for column in ("Nitrogen", "Phosphorus", "Potassium", "Soil_pH")
            ),
            "Weather: °C · % · cm": " · ".join(
                number(record[column])
                for column in ("Temperature", "Humidity", "Rainfall")
            ),
        }

        for rank in (1, 2, 3):

            crop = str(record[f"Top_{rank}_Crop"]).title()

            score = number(record[f"Top_{rank}_Score"], ".2f")

            suffix = "" if score == "-" else "%"

            row[f"Top {rank} crop"] = f"{crop} · {score}{suffix}"

        rows.append(row)

    return pd.DataFrame(rows, index=history.index)


# ==========================================
# 4. SIDEBAR
# ==========================================

inject_css()

if "nav" not in st.session_state:
    st.session_state["nav"] = PAGE_NAMES[0]

if "nav_top" not in st.session_state:
    st.session_state["nav_top"] = st.session_state["nav"]

# Each visitor gets their own empty history list.
if HISTORY_KEY not in st.session_state:
    st.session_state[HISTORY_KEY] = []

with st.sidebar:

    render_html(f"""
        <div class="brand">
            <div class="brand-mark">{icon("sprout", 26)}</div>
            <div>
                <div class="brand-name">AI Crop<br>Recommendation</div>
                <div class="brand-tag">Smart Agriculture</div>
            </div>
        </div>
        """)

    page = st.radio(
        "Navigation",
        PAGE_NAMES,
        key="nav",
        label_visibility="collapsed",
        on_change=sync_from_sidebar,
    )

    render_html("""
        <div class="side-note">
            AI-powered crop recommendations using soil and
            weather parameters.
        </div>
        <div class="side-foot">
            AI Crop Recommendation System<br>
            Python + Scikit-learn + Streamlit
        </div>
        """)


# ==========================================
# 5. LOAD PROJECT FILES
# ==========================================

def stop_with_error(message, detail=None):
    """Show a friendly message and stop. Technical detail goes to the log."""

    st.error(message)

    if detail is not None:
        print(f"[crop-app] {message} :: {type(detail).__name__}: {detail}")

    st.stop()


if not os.path.exists(MODEL_PATH):
    stop_with_error("Trained model file (crop_model.pkl) was not found. Run train_model_v2.py first.")

if not os.path.exists(DATA_PATH):
    stop_with_error("Dataset file (Crop_recommendation.csv) was not found.")

try:
    saved_data = load_model(MODEL_PATH, file_stamp(MODEL_PATH))
except (
    OSError,
    EOFError,
    ValueError,
    AttributeError,
    ImportError,
    pickle.UnpicklingError,
) as error:
    stop_with_error(
        "The model file could not be loaded. It may be damaged or saved with a "
        "different scikit-learn version.",
        error,
    )

try:
    df = load_dataset(DATA_PATH, file_stamp(DATA_PATH))
except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
    stop_with_error("The dataset file could not be read.", error)

if not isinstance(saved_data, dict) or "model" not in saved_data:
    stop_with_error("The model file does not have the expected format (a dictionary with a 'model' key).")

model = saved_data["model"]

if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
    stop_with_error("The saved model cannot give class scores (predict_proba).")

features = saved_data.get("features")

if features is None:
    features = [column for column in df.columns if column != "label"]

if not all(feature in df.columns for feature in features):
    stop_with_error("Model features do not match the dataset columns.")

if len(model.classes_) < 3:
    stop_with_error("The model must contain at least three crop classes.")


model_display_name = get_model_display_name(saved_data, model)

model_accuracy = get_model_accuracy(saved_data)

# Mobile-friendly top menu. Hidden on desktop by CSS (the sidebar is used there).
st.radio(
    "Page",
    PAGE_NAMES,
    key="nav_top",
    horizontal=True,
    label_visibility="collapsed",
    on_change=sync_from_top,
)

crop_category_count = (
    df["label"].nunique() if "label" in df.columns else len(model.classes_)
)

# Crops whose training rows were generated from documented agronomic ranges
# (not field measurements). Read from the model file; empty for older model files.
crop_data_source = saved_data.get("crop_data_source")

if not isinstance(crop_data_source, dict):
    crop_data_source = {}

range_based_crops = sorted(
    str(crop).lower() for crop, kind in crop_data_source.items() if kind != "dataset"
)

range_based_rows = (
    int(df["label"].astype(str).str.lower().isin(range_based_crops).sum())
    if "label" in df.columns and range_based_crops
    else 0
)

range_based_names = ", ".join(crop.title() for crop in range_based_crops)


# ==========================================
# 6. DASHBOARD HOME
# ==========================================

if page == "Dashboard":

    chips = "".join(
        f'<span class="chip">{escape(label)}</span>'
        for label in [
            "Nitrogen",
            "Phosphorus",
            "Potassium",
            "Temperature",
            "Humidity",
            "Soil pH",
            "Rainfall",
        ]
    )

    furrows = "".join(
        f'<path d="M-20 {30 + i * 26} C 120 {i * 26} 220 {64 + i * 26} '
        f'340 {34 + i * 26} S 540 {i * 26} 640 {38 + i * 26}"/>'
        for i in range(12)
    )

    render_html(f"""
        <div class="hero">
            <svg class="hero-art" viewBox="0 0 600 340"
                 preserveAspectRatio="xMidYMid slice" aria-hidden="true">{furrows}</svg>
            <div class="hero-leaf">{icon("leaf", 280)}</div>
            <div class="hero-body">
                <span class="hero-pill">Smart Agriculture Powered by Machine Learning</span>
                <div class="hero-title" role="heading" aria-level="1">AI Crop Recommendation System</div>
                <p class="hero-lede">
                    Enter your soil and weather readings and see the three crops
                    your data matches best, each with a model-estimated score.
                </p>
                <div class="chip-row">{chips}</div>
            </div>
        </div>
        """)

    st.write("")

    st.button(
        "Start a prediction",
        type="primary",
        on_click=go_to_page,
        args=("Crop Prediction",),
    )

    section_heading("Project at a glance")

    stat_cards(
        [
            {
                "icon": "database",
                "label": "Dataset Records",
                "value": f"{len(df):,}",
                "note": (
                    "Labelled soil and weather samples"
                    + (f". Includes {range_based_rows:,} range-based rows" if range_based_rows else "")
                ),
            },
            {
                "icon": "layers",
                "label": "Crop Categories",
                "value": crop_category_count,
                "note": "Crops the model can recommend",
            },
            {
                "icon": "tree",
                "label": "ML Model",
                "value": model_display_name,
                "note": "Trained with Scikit-learn",
                "is_text": True,
            },
            accuracy_stat(saved_data, model_display_name),
        ]
    )

    section_heading(
        "How it works", "Four steps from field readings to a crop recommendation."
    )

    steps = [
        (
            "Enter soil nutrient values",
            "Add nitrogen, phosphorus, potassium and soil pH.",
        ),
        ("Enter weather conditions", "Add temperature, humidity and rainfall."),
        (
            "Machine learning processes the inputs",
            f"A {model_display_name} model compares them with "
            f"{len(df):,} labelled records.",
        ),
        (
            "Receive the top 3 crops",
            "See the three best matches with model-estimated scores.",
        ),
    ]

    render_html('<div class="steps">' + "".join(f"""
            <div class="step">
                <div class="step-num">{number}</div>
                <div class="step-title">{escape(title)}</div>
                <div class="step-text">{escape(text)}</div>
            </div>
            """ for number, (title, text) in enumerate(steps, start=1)) + "</div>")

    section_heading("What you can do")

    feature_items = [
        (
            "trophy",
            "Top 3 recommendations",
            "Ranked crops with a model-estimated score for each one.",
        ),
        (
            "shield",
            "Input range check",
            "Get a warning when a value falls outside the training data.",
        ),
        (
            "bar-chart",
            "Model insights",
            "See which inputs matter most and how the dataset is spread.",
        ),
        (
            "history",
            "Prediction history",
            "Every prediction is saved with its inputs for later review.",
        ),
        (
            "download",
            "CSV export",
            "Download your full prediction history as a CSV file.",
        ),
        (
            "info",
            "Honest scores",
            "Scores are model estimates, not guarantees of farming success.",
        ),
    ]

    render_html('<div class="feature-grid">' + "".join(f"""
            <div class="feature">
                <div class="feature-icon">{icon(icon_name, 22)}</div>
                <div>
                    <div class="feature-title">{escape(title)}</div>
                    <div class="feature-text">{escape(text)}</div>
                </div>
            </div>
            """ for icon_name, title, text in feature_items) + "</div>")


# ==========================================
# 7. CROP PREDICTION
# ==========================================

elif page == "Crop Prediction":

    page_header(
        "Crop Prediction",
        "Enter your farm's soil and weather information to get the "
        "top 3 crop recommendations.",
        "sprout",
    )

    with st.form("crop_prediction_form"):

        col1, col2 = st.columns(2, gap="medium")

        with col1:

            with card_container("card_soil"):

                card_heading(
                    "Soil Information",
                    "Nutrient levels and acidity of your soil",
                    "flask",
                )

                sub1, sub2 = st.columns(2)

                with sub1:

                    N = st.number_input(
                        "Nitrogen (N)",
                        min_value=0.0,
                        max_value=200.0,
                        value=50.0,
                        step=1.0,
                        help="Nitrogen content of the soil. Accepted range: 0 - 200.",
                    )

                with sub2:

                    P = st.number_input(
                        "Phosphorus (P)",
                        min_value=0.0,
                        max_value=150.0,
                        value=50.0,
                        step=1.0,
                        help="Phosphorus content of the soil. Accepted range: 0 - 150.",
                    )

                sub3, sub4 = st.columns(2)

                with sub3:

                    K = st.number_input(
                        "Potassium (K)",
                        min_value=0.0,
                        max_value=210.0,
                        value=50.0,
                        step=1.0,
                        help="Potassium content of the soil. Accepted range: 0 - 210.",
                    )

                with sub4:

                    ph = st.number_input(
                        "Soil pH",
                        min_value=0.0,
                        max_value=14.0,
                        value=6.5,
                        step=0.1,
                        help="Soil acidity on the 0 - 14 pH scale (7 is neutral).",
                    )

        with col2:

            with card_container("card_weather"):

                card_heading(
                    "Weather Information",
                    "Typical conditions for your location",
                    "cloud-sun",
                )

                sub5, sub6 = st.columns(2)

                with sub5:

                    temperature = st.number_input(
                        "Temperature (°C)",
                        min_value=0.0,
                        max_value=50.0,
                        value=25.0,
                        step=1.0,
                        help="Average temperature in °C. Accepted range: 0 - 50.",
                    )

                with sub6:

                    humidity = st.number_input(
                        "Humidity (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=60.0,
                        step=1.0,
                        help="Relative humidity in %. Accepted range: 0 - 100.",
                    )

                rainfall = st.number_input(
                    "Rainfall (cm)",
                    min_value=0.0,
                    max_value=500.0,
                    value=100.0,
                    step=5.0,
                    help=(
                        "Rainfall in centimetres, on the same scale as the training "
                        "data (for example rice about 180 - 300, chickpea about 65 - 95). "
                        "Accepted range: 0 - 500."
                    ),
                )

        st.write("")

        submitted = stretch_call(
            st.form_submit_button, "Predict Top 3 Crops", type="primary"
        )

    # Keep the last submitted inputs so the results stay on screen when
    # Streamlit reruns (for example when the user changes page and comes back,
    # or opens an expander). A new history record is saved only on a real submit.
    if submitted:

        st.session_state["pred_inputs"] = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }

    new_run = bool(submitted)

    input_values = st.session_state.get("pred_inputs")

    if input_values is not None:

        N = input_values["N"]
        P = input_values["P"]
        K = input_values["K"]
        temperature = input_values["temperature"]
        humidity = input_values["humidity"]
        ph = input_values["ph"]
        rainfall = input_values["rainfall"]

        if set(features) != set(input_values):

            st.error("Unexpected model features. Check crop_model.pkl.")

        else:

            input_data = pd.DataFrame(
                [[input_values[feature] for feature in features]], columns=features
            )

            if not np.isfinite(input_data.to_numpy()).all():

                st.error("Please enter valid finite numbers.")

            else:

                # Training dataset range validation
                warnings = []

                for feature in features:
                    minimum = df[feature].min()
                    maximum = df[feature].max()
                    value = input_data[feature].iloc[0]
                    name = FEATURE_LABELS.get(feature, feature)

                    if value < minimum or value > maximum:
                        warnings.append(
                            f"{name}: {value:g} is outside the range seen in "
                            f"the training data ({minimum:.2f} to {maximum:.2f}). "
                            "The model has not seen values like this."
                        )

                # Extra screening checks for the weather and pH inputs.
                # N, P and K are not repeated here: their input boxes already
                # limit them to the same range, so such a check could never fire.
                # These are rough screening limits, not scientific thresholds.
                reliability_checks = {
                    "temperature": (10, 40, "\u00b0C"),
                    "humidity": (20, 100, "%"),
                    "ph": (4, 9, ""),
                    "rainfall": (20, 300, " cm"),
                }

                for feature, (minimum, maximum, unit) in reliability_checks.items():
                    if feature in input_data.columns:
                        value = input_data[feature].iloc[0]
                        name = FEATURE_LABELS.get(feature, feature)

                        if value < minimum or value > maximum:
                            warnings.append(
                                f"{name}: {value:g}{unit} is outside the usual "
                                f"screening range ({minimum}{unit} to {maximum}{unit}). "
                                "Please double-check this measurement."
                            )

                if warnings:

                    callout(
                        "warn",
                        "Predictions may be less reliable.",
                        "alert",
                        items=warnings,
                        title="Unusual input values detected",
                    )

                else:

                    callout(
                        "success",
                        "All inputs are within training dataset ranges.",
                        "check",
                    )

                # Make prediction

                try:

                    probabilities = model.predict_proba(input_data)[0]

                    crop_names = model.classes_

                    top_indices = np.argsort(probabilities)[::-1][:3]

                    top_crops = crop_names[top_indices]

                    top_scores = probabilities[top_indices] * 100

                    # Display results

                    section_heading(
                        "Top 3 recommended crops",
                        "Ranked by the model's estimated score.",
                    )

                    recommendation_cards(top_crops, top_scores)

                    shown_range_based = [
                        str(crop).title()
                        for crop in top_crops
                        if str(crop).lower() in range_based_crops
                    ]

                    if shown_range_based:
                        callout(
                            "info",
                            f"{', '.join(shown_range_based)} "
                            "is based on documented agronomic ranges, not field "
                            "measurements. Treat its score as a guide and check "
                            "with local agricultural advice.",
                            "info",
                        )

                    # Crop recommendation validation
                    section_heading(
                        "Crop recommendation validation",
                        "Your inputs compared with the training data for each crop.",
                    )

                    validation_features = {
                        "N": "Nitrogen (N)",
                        "P": "Phosphorus (P)",
                        "K": "Potassium (K)",
                        "temperature": "Temperature (\u00b0C)",
                        "humidity": "Humidity (%)",
                        "ph": "Soil pH",
                        "rainfall": "Rainfall (cm)",
                    }

                    for position, crop in enumerate(top_crops):

                        crop_data = df[
                            df["label"].astype(str).str.lower() == str(crop).lower()
                        ]

                        if crop_data.empty:
                            continue

                        rows = []
                        passed_checks = 0

                        for feature, display_name in validation_features.items():

                            input_value = float(input_data[feature].iloc[0])
                            minimum = float(crop_data[feature].min())
                            maximum = float(crop_data[feature].max())
                            is_valid = minimum <= input_value <= maximum

                            passed_checks += int(is_valid)

                            rows.append(
                                {
                                    "Parameter": display_name,
                                    "Your input": f"{input_value:g}",
                                    "Observed range": f"{minimum:g} to {maximum:g}",
                                    "Result": "\u2705 Inside" if is_valid else "\u26a0\ufe0f Outside",
                                }
                            )

                        total_checks = len(rows)
                        crop_title = str(crop).title()

                        with st.expander(
                            f"{crop_title}: {passed_checks} of {total_checks} "
                            "inputs inside the observed range",
                            expanded=(position == 0),
                        ):

                            stretch_call(
                                st.dataframe,
                                pd.DataFrame(rows),
                                hide_index=True,
                            )

                    st.caption(
                        "The ranges come from the training dataset only. Being inside "
                        "a range does not prove the crop will grow well, and being "
                        "outside does not prove it will fail."
                    )

                    # This is a score-gap explanation, not calibrated confidence.
                    section_heading(
                        "Prediction confidence analysis",
                        "How far ahead the first crop is from the second.",
                    )

                    top_score = float(top_scores[0])
                    second_score = float(top_scores[1])
                    score_difference = top_score - second_score

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Top Crop Score", f"{top_score:.2f}%")

                    with col2:
                        st.metric("Second Crop Score", f"{second_score:.2f}%")

                    with col3:
                        st.metric(
                            "Score Gap",
                            f"{score_difference:.2f} pts",
                        )

                    if score_difference >= 30:
                        st.success(
                            "The model strongly favours the first recommended crop "
                            "over the second."
                        )
                    elif score_difference >= 10:
                        st.info(
                            "The model moderately favours the first recommended crop."
                        )
                    else:
                        st.warning(
                            "The model's top two crop scores are close. "
                            "Consider comparing both recommendations."
                        )

                    st.caption(
                        "These scores come from the Random Forest's predict_proba output. "
                        "They are model scores, not calibrated probabilities, and they "
                        "are not a guarantee of crop success."
                    )

                    # Save prediction history

                    history_data = {
                        "Date_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Nitrogen": N,
                        "Phosphorus": P,
                        "Potassium": K,
                        "Temperature": temperature,
                        "Humidity": humidity,
                        "Soil_pH": ph,
                        "Rainfall": rainfall,
                        "Top_1_Crop": top_crops[0],
                        "Top_1_Score": round(top_scores[0], 2),
                        "Top_2_Crop": top_crops[1],
                        "Top_2_Score": round(top_scores[1], 2),
                        "Top_3_Crop": top_crops[2],
                        "Top_3_Score": round(top_scores[2], 2),
                    }

                    # Saved only in this visitor's own session (not a shared file),
                    # and only for a new submit (not on every rerun).
                    if new_run:

                        st.session_state[HISTORY_KEY].append(history_data)

                        callout(
                            "success",
                            "Prediction added to your history for this session.",
                            "check",
                        )

                    st.session_state["last_prediction"] = {
                        "input": input_values,
                        "crops": top_crops.tolist(),
                        "scores": top_scores.tolist(),
                    }

                    # Prediction chart

                    chart_df = pd.DataFrame(
                        {"Crop": top_crops, "Model Score (%)": top_scores}
                    )

                    chart_df["Crop"] = chart_df["Crop"].astype(str).str.title()

                    fig = px.bar(
                        chart_df, x="Crop", y="Model Score (%)", text="Model Score (%)"
                    )

                    fig.update_traces(
                        texttemplate="%{text:.2f}%",
                        textposition="outside",
                        marker_color=[GREEN_DARK, GREEN_MID, GREEN_PALE],
                        width=0.5,
                        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                        cliponaxis=False,
                    )

                    style_figure(fig, height=340, bar_radius=8)

                    fig.update_xaxes(
                        title=None,
                        showgrid=False,
                        linecolor=LINE,
                        tickfont=dict(size=14, color=TEXT),
                    )

                    fig.update_yaxes(
                        title="Model score (%)",
                        range=[0, 118],
                        tickvals=[0, 25, 50, 75, 100],
                        gridcolor="#EEF2EE",
                        zeroline=False,
                    )

                    with card_container("chart_prediction"):

                        card_heading(
                            "Top 3 Crop Recommendation Scores",
                            "How strongly the model favours each crop",
                            "bar-chart",
                        )

                        show_chart(fig)

                    callout(
                        "info",
                        "These are model scores, not guaranteed crop-success "
                        "probabilities. "
                        "Check local soil conditions, season and "
                        "agricultural guidance before planting.",
                        "info",
                    )

                except (ValueError, TypeError, KeyError) as error:

                    print(f"[crop-app] Prediction failed :: {type(error).__name__}: {error}")

                    st.error(
                        "The prediction could not be completed. Please check the "
                        "input values and try again."
                    )


# ==========================================
# 8. MODEL INSIGHTS
# ==========================================

elif page == "Model Insights":

    page_header(
        "Model Insights",
        "How the model was built and what it relies on when it recommends a crop.",
        "bar-chart",
    )

    # Dataset and model information

    stat_cards(
        [
            {
                "icon": "database",
                "label": "Dataset Records",
                "value": f"{len(df):,}",
                "note": (
                    "Rows in Crop_recommendation.csv"
                    + (f". Includes {range_based_rows:,} range-based rows" if range_based_rows else "")
                ),
            },
            {
                "icon": "layers",
                "label": "Crop Categories",
                "value": crop_category_count,
                "note": "Distinct crop labels",
            },
            {
                "icon": "hash",
                "label": "Input Features",
                "value": len(features),
                "note": "Soil and weather measurements",
            },
            accuracy_stat(saved_data, model_display_name),
        ]
    )

    # Feature importance

    section_heading(
        "Feature importance",
        "Which inputs the model relies on most, from highest to lowest.",
    )

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame(
            {"Feature": features, "Importance": model.feature_importances_ * 100}
        ).sort_values(by="Importance", ascending=False)

        # Display-only copy with friendly names

        plot_df = importance_df.copy()

        plot_df["Feature"] = plot_df["Feature"].map(
            lambda name: FEATURE_LABELS.get(name, name)
        )

        fig = px.bar(
            plot_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale=[GREEN_PALE, GREEN, GREEN_DARK],
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            cliponaxis=False,
        )

        style_figure(fig, height=380, left_margin=8)

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            bargap=0.35,
        )

        fig.update_xaxes(
            title="Importance (%)", showgrid=True, gridcolor="#EEF2EE", zeroline=False
        )

        fig.update_yaxes(title=None, showgrid=False, tickfont=dict(size=14, color=TEXT))

        chart_col, table_col = st.columns([3, 2], gap="medium")

        with chart_col:

            with card_container("chart_importance"):

                card_heading(
                    "Overall Random Forest Feature Importance",
                    "Share of the model's decisions by input",
                    "bar-chart",
                )

                show_chart(fig)

        with table_col:

            with card_container("panel_importance"):

                card_heading(
                    "Importance values", "Sorted from highest to lowest", "clipboard"
                )

                stretch_call(
                    st.dataframe,
                    plot_df,
                    hide_index=True,
                    column_config={
                        "Feature": st.column_config.TextColumn("Feature"),
                        "Importance": st.column_config.NumberColumn(
                            "Importance (%)", format="%.2f"
                        ),
                    },
                )

        callout(
            "info",
            "Feature importance describes overall model behaviour, "
            "not the explanation for an individual prediction.",
            "info",
        )

    else:

        callout("info", "Feature importance is not available for this model.", "info")

    # Crop distribution

    if "label" in df.columns:

        section_heading(
            "Crop distribution", "Number of dataset records for each crop category."
        )

        crop_counts = df["label"].value_counts().reset_index()

        crop_counts.columns = ["Crop", "Records"]

        crop_counts["Crop"] = crop_counts["Crop"].astype(str).str.title()

        fig = px.bar(
            crop_counts, x="Records", y="Crop", orientation="h", text="Records"
        )

        fig.update_traces(
            marker_color=GREEN,
            textposition="outside",
            hovertemplate="%{y}: %{x} records<extra></extra>",
            cliponaxis=False,
        )

        style_figure(fig, height=max(420, 26 * len(crop_counts) + 40), bar_radius=4)

        fig.update_layout(
            yaxis={"categoryorder": "total ascending", "dtick": 1}, bargap=0.3
        )

        fig.update_xaxes(
            title="Records", showgrid=True, gridcolor="#EEF2EE", zeroline=False
        )

        fig.update_yaxes(title=None, showgrid=False, tickfont=dict(size=13, color=TEXT))

        with card_container("chart_distribution"):

            card_heading(
                "Number of Records Per Crop",
                f"All {len(crop_counts)} crop categories in the dataset",
                "layers",
            )

            show_chart(fig)

            if range_based_crops:
                st.caption(
                    f"Range-based crops ({range_based_names}) were generated from "
                    "documented agronomic ranges, not measured in the field. "
                    "The other crops come from the original dataset."
                )

    # Dataset statistics

    section_heading(
        "Dataset statistics",
        "Minimum, average and maximum of each input in the training data.",
    )

    summary = df[features].agg(["min", "mean", "max"]).T

    summary.index = [FEATURE_LABELS.get(name, name) for name in summary.index]

    summary.index.name = "Feature"

    summary = summary.reset_index()

    summary.columns = ["Feature", "Minimum", "Average", "Maximum"]

    with card_container("panel_dataset"):

        card_heading(
            "Training data ranges",
            "Values outside these ranges trigger a reliability warning",
            "database",
        )

        stretch_call(
            st.dataframe,
            summary,
            hide_index=True,
            column_config={
                "Minimum": st.column_config.NumberColumn(format="%.2f"),
                "Average": st.column_config.NumberColumn(format="%.2f"),
                "Maximum": st.column_config.NumberColumn(format="%.2f"),
            },
        )


# ==========================================
# 9. PREDICTION HISTORY
# ==========================================

elif page == "Prediction History":

    page_header(
        "Prediction History",
        "Your predictions from this visit. Only you can see them.",
        "history",
    )

    # Only this visitor's own records (kept in st.session_state)
    history = pd.DataFrame(st.session_state[HISTORY_KEY])

    if history.empty:

        callout(
            "info",
            "No predictions yet in this session. "
            "Make a prediction and it will show up here.",
            "info",
        )

        st.button(
            "Make a prediction",
            type="primary",
            on_click=go_to_page,
            args=("Crop Prediction",),
        )

    else:

        stat_cards(
            [
                {
                    "icon": "hash",
                    "label": "Total Predictions",
                    "value": f"{len(history):,}",
                    "note": "In this session",
                },
                {
                    "icon": "layers",
                    "label": "Unique Top Crops",
                    "value": history["Top_1_Crop"].nunique(),
                    "note": "Different first-choice crops",
                },
                {
                    "icon": "star",
                    "label": "Most Recommended Crop",
                    "value": str(history["Top_1_Crop"].mode().iloc[0]).title(),
                    "note": "Most frequent first choice",
                    "is_text": True,
                },
            ]
        )

        section_heading(
            "Prediction records",
            "Newest first. The download has your records from this session only.",
        )

        # Column labels for the table (only for columns that exist)

        labels = {
            "Date_Time": "Date & Time",
            "Nitrogen": "Nitrogen",
            "Phosphorus": "Phosphorus",
            "Potassium": "Potassium",
            "Temperature": "Temp °C",
            "Humidity": "Humidity %",
            "Soil_pH": "Soil pH",
            "Rainfall": "Rain cm",
            "Top_1_Crop": "Top 1 crop",
            "Top_1_Score": "Top 1 %",
            "Top_2_Crop": "Top 2 crop",
            "Top_2_Score": "Top 2 %",
            "Top_3_Crop": "Top 3 crop",
            "Top_3_Score": "Top 3 %",
        }

        column_config = {
            column: st.column_config.Column(label)
            for column, label in labels.items()
            if column in history.columns
        }

        for column in ("Top_1_Score", "Top_2_Score", "Top_3_Score"):

            if column in history.columns:

                column_config[column] = st.column_config.NumberColumn(
                    labels[column], format="%.2f"
                )

        compact_history = build_compact_history(history)

        with card_container("panel_history"):

            show_all_columns = True

            if compact_history is not None:

                # st.toggle needs a recent Streamlit; fall back to a checkbox

                switch = getattr(st, "toggle", st.checkbox)

                show_all_columns = switch(
                    "Show every column",
                    value=False,
                    key="history_all_columns",
                    help="Switch to the full table with every input "
                    "and score in its own column.",
                )

            # Display order only: newest record first

            if show_all_columns:

                stretch_call(
                    st.dataframe,
                    history.iloc[::-1],
                    hide_index=True,
                    column_config=column_config,
                )

            else:

                stretch_call(
                    st.dataframe, compact_history.iloc[::-1], hide_index=True
                )

            # Download CSV

            csv_data = history.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download my history (CSV)",
                data=csv_data,
                file_name="my_prediction_history.csv",
                mime="text/csv",
            )

        # History chart

        if "Top_1_Crop" in history.columns:

            section_heading(
                "Most frequently recommended crops",
                "How often each crop was the first choice.",
            )

            crop_counts = history["Top_1_Crop"].value_counts().reset_index()

            crop_counts.columns = ["Crop", "Count"]

            crop_counts["Crop"] = crop_counts["Crop"].astype(str).str.title()

            fig = px.bar(
                crop_counts, x="Count", y="Crop", orientation="h", text="Count"
            )

            fig.update_traces(
                marker_color=GREEN,
                textposition="outside",
                hovertemplate="%{y}: %{x} predictions<extra></extra>",
                cliponaxis=False,
            )

            style_figure(fig, height=max(260, 46 * len(crop_counts) + 60))

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}, bargap=0.4
            )

            fig.update_xaxes(
                title="Times recommended first",
                showgrid=True,
                gridcolor="#EEF2EE",
                zeroline=False,
                dtick=1,
            )

            fig.update_yaxes(
                title=None, showgrid=False, tickfont=dict(size=14, color=TEXT)
            )

            with card_container("chart_history"):

                card_heading(
                    "Top Recommendation Frequency",
                    "Based on your saved predictions",
                    "bar-chart",
                )

                show_chart(fig)
