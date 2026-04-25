"""
STGNN Pollution Dashboard — Main App
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, sys

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STGNN Pollution Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hide Streamlit auto-generated multipage nav ───────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebarNav"]          { display: none !important; }
[data-testid="stSidebarNavItems"]      { display: none !important; }
section[data-testid="stSidebar"] > div:first-child > div:first-child ul
                                       { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── CSS injection ─────────────────────────────────────────────────────────────
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from preprocessing.data_cleaning import load_raw_data, clean_data
from models.forecast_model import forecast_all_cities
from dashboard.pages import overview, city_analysis, monthly_trends, predictions, risk_map


# ── Data Loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    raw = load_raw_data()
    df = clean_data(raw)
    return df


@st.cache_data(show_spinner=False)
def load_forecasts(df_hash):
    df = load_data()
    return forecast_all_cities(df, months_ahead=90)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:10px 0 20px 0;'>
            <div style='font-size:40px;'>🌫️</div>
            <div style='font-size:18px;font-weight:bold;color:#00d4ff;'>STGNN Dashboard</div>
            <div style='font-size:11px;color:#64748b;margin-top:4px;'>Spatiotemporal Graph Neural Network</div>
            <div style='font-size:11px;color:#64748b;'>India Air Quality · 2015–2027</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        page = st.radio(
            "📌 Navigation",
            ["🌐 Overview", "🏙️ City Analysis", "📅 Monthly Trends", "🔮 Predictions", "⚠️ Risk Map"],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("""
        <div style='font-size:12px;color:#64748b;'>
        <b style='color:#94a3b8;'>📊 Dataset Info</b><br>
        • 26 Indian cities<br>
        • Daily readings 2015–2020<br>
        • 12 pollutant features<br>
        • ~29,000 records<br><br>
        <b style='color:#94a3b8;'>🤖 Model</b><br>
        • Spatial GCN layers<br>
        • Temporal GRU encoder<br>
        • Graph-neighbour blending<br>
        • Seasonal decomposition<br>
        • Forecast horizon: 2027
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""
        <div style='font-size:11px;color:#475569;text-align:center;'>
        AQI Scale Reference<br>
        🟢 Good ≤50 · 🟡 Satisfactory ≤100<br>
        🟠 Moderate ≤200 · 🔴 Poor ≤300<br>
        🔴 Very Poor ≤400 · 🟣 Severe >400
        </div>
        """, unsafe_allow_html=True)

    return page


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    page = render_sidebar()

    with st.spinner("⚙️ Loading and cleaning pollution data..."):
        df = load_data()

    with st.spinner("🔮 Running STGNN forecast model..."):
        try:
            forecast_df = load_forecasts(len(df))
        except Exception as e:
            st.warning(f"Forecast model warning: {e}")
            forecast_df = pd.DataFrame()

    if page == "🌐 Overview":
        overview.show(df, forecast_df)
    elif page == "🏙️ City Analysis":
        city_analysis.show(df, forecast_df)
    elif page == "📅 Monthly Trends":
        monthly_trends.show(df, forecast_df)
    elif page == "🔮 Predictions":
        predictions.show(df, forecast_df)
    elif page == "⚠️ Risk Map":
        risk_map.show(df, forecast_df)


if __name__ == "__main__":
    main()