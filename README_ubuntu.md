````markdown
# Argus Realtime v7.0 🌍🌫️ (Ubuntu 22.04 Setup Guide)

Welcome to **ARGUS v7.0**, a real-time smog forecasting system powered by an ensemble of XGBoost, LightGBM, CatBoost, and PyTorch (BiGRU), trained on NASA satellite data.

This guide provides **clean, reproducible steps for Ubuntu 22.04**.

---

## 🚀 Quick Start

### 1. Install System Dependencies

Ensure Python 3.10+ and required tools are installed:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libhdf4-dev libhdf4-0
````

---

### 2. Navigate to Project Directory

```bash
cd ~/Downloads/Realtime_v7.0
```

(Adjust path if different)

---

### 3. Create and Activate Virtual Environment (IMPORTANT)

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Upgrade pip and build tools

```bash
pip install --upgrade pip setuptools wheel
```

---

### 5. Install Core Scientific Stack (Prevent NumPy Errors)

```bash
pip install numpy==1.26.4 scipy==1.11.4
```

---

### 6. Install Project Requirements

```bash
pip install -r requirements.txt
```

> ⚠️ Do NOT mix system packages (`apt`) with `pip` installs for Python libraries like NumPy/SciPy.

---

### 7. Verify Installation

```bash
python3 -c "import numpy; import scipy; import xgboost; print('OK')"
```

If you see `OK`, everything is correctly installed.

---

## 🔐 Authentication

### Google Earth Engine

```bash
earthengine authenticate
```

Follow the browser login process.

---

### NASA Earthdata (LANCE App Key)

* Obtain your **NASA LANCE App Key** from Earthdata
* Enter it in the app sidebar after launch
* It will be saved locally in a `.env` file

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The app will open in your browser.

---

## 📊 Using the App

1. Set **Latitude & Longitude** (default: Lahore `31.5204, 74.3587`)
2. (Optional) Enter NASA LANCE App Key
3. Confirm all models show `✅ Success`
4. Click **🛰️ Fetch Real-Time Data & Forecast**

---

## ⚠️ Troubleshooting

### ❌ `numpy.core.multiarray failed to import`

Cause: Mixed system + pip installations

Fix:

```bash
pip uninstall -y numpy scipy xgboost
rm -rf ~/.local/lib/python3.10/site-packages/numpy*
rm -rf ~/.local/lib/python3.10/site-packages/scipy*

# Recreate environment
python3 -m venv venv
source venv/bin/activate

pip install numpy==1.26.4 scipy==1.11.4
pip install -r requirements.txt
```

---

### ❌ `pip not found`

```bash
sudo apt install python3-pip
```

---

### ❌ `pyhdf` install error

```bash
sudo apt install libhdf4-dev
```

---

### ❌ `earthengine: command not found`

```bash
pip install earthengine-api
```

---

### ❌ Permission errors

Avoid using `sudo pip`. Always use a virtual environment.

---

## 🧠 Notes

* Ubuntu fully supports `pyhdf` (unlike native Windows)
* Direct access to NASA LANCE NRT (~0–6 hour latency)
* No WSL required

---

## ✅ Best Practice Summary

Always run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Never mix:

* `apt install python3-scipy`
* with `pip install numpy`

---

## 📌 You're Ready

If setup completes without errors, launch the app and start generating real-time smog forecasts.

For issues, check the **first error in logs** and resolve dependencies accordingly.

```
```

