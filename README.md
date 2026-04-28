# Argus Realtime v7.0 🌍🌫️

Welcome to the ARGUS v7.0 Real-time Smog Forecasting Application. This system utilizes a deep ensemble of XGBoost, LightGBM, CatBoost, and PyTorch (BiGRU) trained on NASA satellite metrics to deliver real-time T+0 and T+1 smog forecasts.

## 🚀 Quick Start Guide

Follow these steps to configure, install, and run the Argus prediction application natively on your system.

### 1. Requirements & Dependencies

This system requires python 3.10+ and relies on several significant mathematical libraries. Open your terminal or powershell and navigate to the root directory where the application is stored:

```powershell
cd "d:\Antigravity Projects\Realtime_v7.0"
```

Next, install all the requisite backend technologies using `pip`:

```powershell
pip install -r requirements.txt
```

> **Note**: This will install bulky packages including `torch` (for the BiGRU sequence), `pyhdf` (for processing NASA satellite structures), and standard boosting frameworks (`xgboost`, `catboost`, `lightgbm`). This might take a few minutes.

#### Windows users: pyhdf and LANCE NRT

`pyhdf` has no pre-built Windows wheel for Python 3.10+, so direct installation of NASA LANCE HDF4 files is only supported on Linux. The fetcher now handles both paths:

- **WSL2 (recommended on Windows 11)** — gives full LANCE NRT with the 5×5 pixel patch used during v7.0 training, ~0–6 h AOD lag:
   ```powershell
   # Once, in PowerShell (Admin)
   wsl --install
   ```
   Then inside the WSL Ubuntu shell:
   ```bash
   sudo apt update && sudo apt install -y libhdf4-dev libhdf4-0-dev python3-pip
   cd /mnt/c/Users/Dell/Downloads/Realtime_v7.0
   pip install -r requirements.txt
   earthengine authenticate
   streamlit run app.py
   ```
- **Native Windows (no WSL2)** — `pyhdf` import will fail gracefully; the fetcher then falls through to the **ORNL MODIS JSON API** (single nearest-pixel AOD at ~3–6 h lag), then MERRA-2 (~1 d), MAIAC-GEE (40–60 d last resort), and Open-Meteo CAMS. Your **NASA App Key** is still used for both the pyhdf and JSON API paths.

### 2. Authentication

ARGUS v7.0 retrieves live atmospheric sequences from two distinct cloud systems: Google Earth Engine (GEE) and NASA LANCE NRT. 

#### Google Earth Engine
Before running the application, ensure that you have an active authentication context locally to allow python to ping `earthengine` endpoints natively:

```powershell
earthengine authenticate
```
*Follow the browser prompts that open to sync your Google cloud project.*

#### NASA Earthdata (LANCE App Key)
To utilize the ultra-low latency AOD extraction you must possess a NASA LANCE _App Key_. You can acquire this securely from your NASA Earthdata portal. 
* Once the application launches, you will simply enter the key into the Sidebar on the left-hand side and click **"Save Key"**. 
* The interface utilizes `python-dotenv` to generate a local `.env` configuration file to save it without compromising the source code.

### 3. Launching the App

To boot the interface over your local browser, run Streamlit:

```powershell
streamlit run app.py
```

### 4. Fetching the Forecasts

1. Select your target **Latitude** and **Longitude** within the sidebar (defaults to Lahore `31.5204, 74.3587`).
2. Optional: Provide your NASA LANCE NRT Key. If excluded, the fetcher will cleanly fall back to `MAIAC-GEE` latency data.
3. Observe the **Model Load Status**. Make sure all 6 artifacts resolve as `✅ Success`.
4. Click `🛰️ Fetch Real-Time Data & Forecast` to begin orchestration. The system will buffer 35 days of history from Open-Meteo, correlate the latest S5P / VIIRS satellite tiles, evaluate predictions using conformal arrays, and output AQI natively!
