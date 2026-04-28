# ARGUS v7.0 Deployment Guide

This guide covers how to deploy the Argus v7.0 Real-time Smog Forecasting Dashboard locally on Windows and Ubuntu, as well as how to deploy it to the cloud.

---

## 1. Ubuntu Deployment (Recommended for Local)

Ubuntu natively supports all scientific libraries required by this project, including `pyhdf` which is necessary for downloading live NASA LANCE NRT satellite data.

### Step-by-Step Setup
1. **Install Virtual Environment Package:**
   Debian/Ubuntu requires the `venv` package to be explicitly installed to avoid `ensurepip` errors.
   ```bash
   sudo apt update
   sudo apt install python3.10-venv
   ```
2. **Create the Virtual Environment:**
   ```bash
   python3 -m venv venv
   ```
3. **Activate and Install:**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Google Earth Engine Auth:**
   ```bash
   earthengine authenticate
   ```
5. **Run the App:**
   ```bash
   streamlit run app.py
   ```

---

## 2. Windows Deployment

> [!WARNING]
> **Native Windows is NOT recommended** because the `pyhdf` library (required for NASA LANCE NRT data) does not easily compile on Windows. It is highly recommended to use **WSL (Windows Subsystem for Linux)**.

### Option A: Using WSL (Recommended)
1. Open PowerShell and install WSL:
   ```powershell
   wsl --install
   ```
2. Restart your computer, open the **Ubuntu** terminal from your Start menu, and follow the **Ubuntu Deployment** instructions above.

### Option B: Native Windows (Advanced)
If you *must* use native Windows, you will need to download an unofficial pre-compiled wheel for `pyhdf` from Christoph Gohlke's archive, or the application will silently fall back to slower GEE/MERRA-2 data.

1. **Create the Virtual Environment:**
   ```cmd
   python -m venv venv
   ```
2. **Activate the Environment:**
   ```cmd
   venv\Scripts\activate
   ```
3. **Install Requirements:**
   ```cmd
   pip install -r requirements.txt
   ```
4. **Run the App:**
   ```cmd
   streamlit run app.py
   ```

---

## 3. Cloud Deployment (Streamlit Community Cloud)

When deploying to a public cloud environment like Streamlit Community Cloud, AWS, or Heroku, you should **never** upload your local `.env` file containing your NASA App Key.

Instead, Argus v7.0 is configured to use **Streamlit Secrets**.

### Steps for Streamlit Community Cloud:
1. Push this repository to GitHub (ensure `.env` is in your `.gitignore`).
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/) and click **New app**.
3. Select your GitHub repository, branch, and set the Main file path to `app.py`.
4. **Before clicking Deploy**, click on **Advanced settings**.
5. Under the **Secrets** text box, paste your NASA App Key in TOML format:
   ```toml
   NASA_APP_KEY = "your_long_nasa_token_here"
   ```
6. Click **Save** and then **Deploy!**

### Google Earth Engine in the Cloud:
If you are deploying to a server, `earthengine authenticate` (which opens a browser) will not work. You will need to use a **Google Cloud Service Account**.

1. **Create a Service Account:**
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Select your project (`dazzling-trail-479218-p2`).
   - Navigate to **IAM & Admin** > **Service Accounts** and click **Create Service Account**.
   - Grant it the `Earth Engine Resource Viewer` role.
   - Click on the new service account, go to the **Keys** tab, click **Add Key** > **Create new key** > **JSON**. This will download a `.json` file to your computer.

2. **Add the Key to Streamlit Secrets:**
   - Open the downloaded `.json` file and copy its entire contents.
   - In your Streamlit Community Cloud app settings, paste it under the **Secrets** text box using the key `EARTHENGINE_TOKEN`:
     ```toml
     NASA_APP_KEY = "your_long_nasa_token_here"
     
     [EARTHENGINE_TOKEN]
     type = "service_account"
     project_id = "dazzling-trail-..."
     private_key_id = "..."
     private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
     client_email = "..."
     client_id = "..."
     auth_uri = "https://accounts.google.com/o/oauth2/auth"
     token_uri = "https://oauth2.googleapis.com/token"
     auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
     client_x509_cert_url = "..."
     ```
   *(Ensure you match the exact structure of your downloaded JSON file into TOML format, or pass it as a JSON string literal).*

3. **Modify `app.py` for Cloud Initialization (Optional):**
   The application uses the `ee.Authenticate()` and `ee.Initialize()` methods. If you set up the `EARTHENGINE_TOKEN` secret as shown above, you can modify the Earth Engine initialization block in `earth_engine_fetcher.py` to use those credentials automatically:
   
   ```python
   import streamlit as st
   
   try:
       # For local runs
       ee.Initialize(project='dazzling-trail-479218-p2')
   except Exception:
       try:
           # For Cloud Deployment using st.secrets
           credentials = ee.ServiceAccountCredentials(
               st.secrets["EARTHENGINE_TOKEN"]["client_email"],
               key_data=st.secrets["EARTHENGINE_TOKEN"]["private_key"]
           )
           ee.Initialize(credentials, project='dazzling-trail-479218-p2')
       except Exception:
           # Fallback for local auth
           ee.Authenticate()
           ee.Initialize(project='dazzling-trail-479218-p2')
   ```
