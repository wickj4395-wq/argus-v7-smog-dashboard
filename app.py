import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from prediction import SmogPredictor
from earth_engine_fetcher import EarthEngineDataFetcher
from preprocessing import aqi_to_smog_level
from dotenv import load_dotenv, set_key
import os

load_dotenv()

st.set_page_config(
    page_title="Argus v7.0: Smog Forecast",
    page_icon="🌫",
    layout="wide",
    initial_sidebar_state="expanded"
)

CITIES = {
    "Lahore":     (31.5204, 74.3587),
    "Islamabad":  (33.6844, 73.0479),
    "Karachi":    (24.8607, 67.0011),
    "Faisalabad": (31.4504, 73.1350),
    "Multan":     (30.1575, 71.5249),
    "Peshawar":   (34.0151, 71.5249),
}

@st.cache_resource
def load_predictor():
    return SmogPredictor()

def aqi_action(aqi):
    """Return (icon_key, short_label, card_color) for the health card."""
    if aqi <= 100: return "check", "Safe to go outside",   "#2E7D32"
    if aqi <= 150: return "mask",  "Wear a mask outdoors", "#E65100"
    if aqi <= 200: return "warn",  "Limit time outside",   "#B71C1C"
    return                "home",  "Stay indoors",          "#6A1B9A"

ACTION_ICONS = {
    "check": "&#9989;",
    "mask":  "&#128567;",
    "warn":  "&#9888;",
    "home":  "&#127968;",
}

def driver_buckets(fc_row, fire, gas):
    """Heuristic source attribution from fetched features."""
    fire_s = min(fire.get("fire_india", 0) + fire.get("fire_pak", 0), 300) / 300
    traf_s = min((gas.get("no2", 0) / 3e-4 + gas.get("co", 0) / 0.1) / 2, 1.0)
    ae     = fc_row.get("angstrom_exp", 1.0) or 1.0
    dust_s = max(0.0, 1.0 - ae)
    blh    = fc_row.get("blh_min", 500) or 500
    inv_s  = 1.0 if blh < 200 else 0.2
    total  = fire_s + traf_s + dust_s + inv_s + 0.01
    return {
        "Fires":             round(fire_s / total * 100),
        "Traffic/Industry":  round(traf_s / total * 100),
        "Dust":              round(dust_s / total * 100),
        "Inversion/Weather": round(inv_s  / total * 100),
    }

def main():
    st.title("Argus v7.0 — Smog Forecasting")
    st.caption("NASA LANCE NRT · Sentinel-5P · Open-Meteo")

    with st.spinner("Loading ensemble models..."):
        predictor = load_predictor()

    # Sidebar
    st.sidebar.header("Configuration")

    current_key = ""
    try:
        current_key = st.secrets.get("NASA_APP_KEY", "")
    except Exception:
        pass
    if not current_key:
        current_key = os.getenv("NASA_APP_KEY", "")

    nasa_key = st.sidebar.text_input(
        "NASA App Key (LANCE NRT)", value=current_key, type="password",
        help="Required for fresh AOD data. Falls back to MERRA-2 then GEE if missing."
    )
    if st.sidebar.button("Save Key"):
        set_key(".env", "NASA_APP_KEY", nasa_key)
        st.sidebar.success("Key saved to local .env (For cloud deploy, configure Streamlit Secrets instead)")

    with st.sidebar.expander("Model Load Status", expanded=True):
        for name, msg in predictor.get_load_status().items():
            ok = "Success" in msg
            st.write(("✅ " if ok else "❌ ") + name + ("" if ok else ": " + msg))

    # Location picker
    st.sidebar.header("Location")
    city = st.sidebar.selectbox("Quick-select city", ["Custom"] + list(CITIES.keys()))
    default_lat, default_lon = CITIES.get(city, (31.5204, 74.3587))

    m = folium.Map(location=[default_lat, default_lon], zoom_start=7,
                   tiles="CartoDB positron")
    for cname, (clat, clon) in CITIES.items():
        folium.Marker(
            [clat, clon], tooltip=cname,
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
    map_data = st_folium(m, height=300, width=None, key="loc_map")

    if map_data and map_data.get("last_object_clicked"):
        lat = map_data["last_object_clicked"]["lat"]
        lon = map_data["last_object_clicked"]["lng"]
    elif map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
    else:
        lat, lon = default_lat, default_lon

    st.sidebar.caption(f"Active: {lat:.4f}, {lon:.4f}")

    if not st.sidebar.button("Fetch & Forecast", type="primary"):
        st.info("Select a location on the map or choose a city, then click Fetch & Forecast.")
        return

    # Fetch
    fetcher = EarthEngineDataFetcher(nasa_key=nasa_key, lat=lat, lon=lon)
    with st.spinner("Fetching satellite, gas, fire, and weather data (up to 60 s)..."):
        try:
            hist_rows, fc_rows, mdata, gas, fire = fetcher.fetch_all_data()
        except Exception as e:
            st.error(f"Fetch error: {e}")
            return

    if not hist_rows:
        st.error("No historical data assembled. Check Earth Engine auth.")
        return
    if not mdata or "aod_source" not in mdata:
        st.error("AOD metadata missing — all satellite sources failed.")
        return

    lag_days = (datetime.utcnow().date() - pd.Timestamp(hist_rows[-1]["date"]).date()).days

    if lag_days > 3:
        st.warning(
            f"⚠️ AOD data is **{lag_days} days old** (source: {mdata['aod_source']}). "
            "Predictions may be less accurate. "
            "Install pyhdf via WSL2 for fresh LANCE NRT data."
        )
    else:
        st.success(f"Data fetched · **{mdata['aod_source']}** · lag {lag_days}d")

    # Raw satellite parameters (collapsed)
    with st.expander("Raw Satellite Parameters", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AOD 047",       f"{mdata['modis_aod']:.3f}",         f"src: {mdata['aod_source']}")
        c2.metric("Angstrom Exp",  f"{mdata['angstrom_exp']:.2f}")
        c3.metric("NO₂",           f"{gas.get('no2', 0):.5f}")
        c4.metric("Fires PK+IND",  f"{fire.get('fire_pak',0)+fire.get('fire_india',0):.0f}")
        c5.metric("UVAI",          f"{gas.get('uvai', 0):.2f}")

    # Predict
    with st.spinner("Running ensemble prediction..."):
        try:
            predictions = predictor.predict_forecasts(hist_rows, fc_rows)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

    if not predictions:
        st.error("No predictions could be generated. This may be due to missing weather forecast data.")
        return

    # Health action cards
    st.subheader("Forecast")
    cols = st.columns(len(predictions))
    for idx, pred in enumerate(predictions):
        sl, color, adv, activity = aqi_to_smog_level(pred["aqi"])
        icon_key, action_label, card_color = aqi_action(pred["aqi"])
        icon_html = ACTION_ICONS[icon_key]
        with cols[idx]:
            st.markdown(f"""
<div style="padding:1rem;border-radius:12px;background:{card_color};
text-align:center;color:white;margin-bottom:0.5rem">
<div style="font-size:2rem">{icon_html}</div>
<div style="font-size:0.95rem;font-weight:600;margin-top:4px">{action_label}</div>
<div style="font-size:0.75rem;margin-top:2px;opacity:0.85">
    {pred["label"]} &bull; {pred["date"]}
</div>
<div style="font-size:1.6rem;font-weight:700;margin-top:8px">
    AQI {pred["aqi"]:.0f}
</div>
<div style="font-size:0.75rem">{pred["aqi_category"]}</div>
</div>""", unsafe_allow_html=True)
            with st.expander("Details"):
                st.write(f"90% CI: [{pred['ci_low']:.0f} to {pred['ci_high']:.0f}]")
                st.write(f"Smog level: {sl}")
                st.info(adv)
                st.write("Activity guide:")
                act_cols = st.columns(5)
                labels   = ["Walking", "Cycling", "School", "Elderly", "Asthmatic"]
                keys     = ["walking", "cycling", "school", "elderly", "asthmatic"]
                for ac, lbl, key in zip(act_cols, labels, keys):
                    ok = activity.get(key, False)
                    ac.metric(lbl, "OK" if ok else "Avoid")

    # Driver attribution bar
    if fc_rows:
        buckets = driver_buckets(fc_rows[0], fire, gas)
        fig_d = go.Figure(go.Bar(
            x=list(buckets.values()),
            y=list(buckets.keys()),
            orientation="h",
            marker_color=["#C62828", "#EF6C00", "#F9A825", "#1565C0"],
            text=[f"{v}%" for v in buckets.values()],
            textposition="inside",
        ))
        fig_d.update_layout(
            title="Estimated smog sources today (heuristic)",
            xaxis_title="Contribution (%)",
            template="plotly_white",
            height=220,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_d, use_container_width=True)
        st.caption(
            "Heuristic estimate from fire counts, NO₂/CO, Angstrom exponent, and BLH. "
            "SHAP-based attribution is planned for a future release."
        )

    # AQI bar chart with CI
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="AQI Forecast",
        x=[p["label"] for p in predictions],
        y=[p["aqi"]   for p in predictions],
        error_y=dict(
            type="data", symmetric=False,
            array=[p["ci_high"] - p["aqi"] for p in predictions],
            arrayminus=[p["aqi"] - p["ci_low"] for p in predictions]
        ),
        marker_color="royalblue"
    ))
    fig.add_hline(y=150, line_dash="dash", line_color="red",    annotation_text="Unhealthy")
    fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="USG")
    fig.update_layout(
        title="AQI Forecast with 90% Confidence Intervals",
        yaxis_title="US EPA AQI",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
