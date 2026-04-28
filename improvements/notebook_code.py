# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ARGUS v7.0 — Smog Detection & AQI Forecasting for Lahore, Pakistan          ║
# ║  NUST-SEECS | GEE: dazzling-trail-479218-p2                                  ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  BUGS FIXED vs v6.0 (from actual v6.0 results analysis):                     ║
# ║  [BUG-1] Broken hyperparameter remap via chained .replace():                 ║
# ║    XGBoost: 'lr' -> 'learning_reg_alphate' (learning_rate contains 'ra')    ║
# ║    LightGBM: 'lr' same, 'nl'->'n_estimatorsl', 'mcs'->'mcolsample_bytree'   ║
# ║    FIX: Explicit KEY_MAP dicts. All Optuna-tuned params now used correctly.  ║
# ║  [BUG-2] AOD Cell 13 uses eval_set=test set -> test-set leakage + overfit   ║
# ║    FIX: Use 15% holdout from training set for AOD early stopping.             ║
# ║  [BUG-3] Walk-forward starts 2019: 3yr training -> constant prediction R2=0 ║
# ║    FIX: Walk-forward starts 2022 (6yr training minimum). 85/15 split.        ║
# ║  [BUG-4] Live MAIAC: GEE lag >60d. LANCE NRT primary (3-6hr latency).       ║
# ║    FIX: fetch_maiac_lance_nrt() as primary; GEE + MERRA-2 as fallbacks.     ║
# ║  [BUG-5] BiGRU R2=0.042: seq_len=21 too long, no LR warmup.                 ║
# ║    FIX: GRU_SEQ_LEN=14, 5-epoch linear LR warmup added.                     ║
# ║  STORAGE RESTRUCTURE:                                                         ║
# ║    PERM_DIR=v7_0/ (models, results, forecasts, optuna) -- NEVER DELETE       ║
# ║    TEMP_DIR=v7_0_temp/ (caches, mid-training checkpoints) -- OK to delete   ║
# ║  NEW FEATURES (8):                                                            ║
# ║    aod_std_30, aod_trend_7/14, wind_india_3d/7d,                            ║
# ║    monsoon_flag, heat_wave, dust_flag                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
!pip install rasterio earthengine-api xgboost lightgbm catboost torch scikit-learn optuna shap pyhdf --quiet

from google.colab import drive
drive.mount('/content/drive')

import calendar, os, json, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns, requests, joblib, warnings, ee, optuna, shap
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import rasterio

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42); tf.random.set_seed(42); torch.manual_seed(42)

GEE_PROJECT = 'dazzling-trail-479218-p2'
try:
    ee.Initialize(project=GEE_PROJECT)
    print(f'✅ Earth Engine: {GEE_PROJECT}')
except Exception:
    ee.Authenticate(); ee.Initialize(project=GEE_PROJECT)
print('✅ Argus v7.0 imports done')

# === Cell 2: Config v7.0 ===
# STORAGE RESTRUCTURE: permanent vs temporary clearly separated
class Config:
    GEE_PROJECT  = 'dazzling-trail-479218-p2'
    DATA_DIR     = Path('/content/drive/MyDrive/Argus_Data')

    # 🛑 FIX: Add your NASA Earthdata App Key here!
    NASA_APP_KEY = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJlbWFpbF9hZGRyZXNzIjoid2lja2o0Mzk1QGdtYWlsLmNvbSIsImlzcyI6IkFQUyBPQXV0aDIgQXV0aGVudGljYXRvciIsImlhdCI6MTc3NTYyNTE0NiwibmJmIjoxNzc1NjI1MTQ2LCJleHAiOjE5MzMzMDUxNDYsInVpZCI6ImFsaV90YWltb29yX2toYWxpZCIsInRva2VuQ3JlYXRvciI6ImFsaV90YWltb29yX2toYWxpZCJ9.H540TXN_Kk2s27h_MoLuCiLh7trZC4Lha3F2LXDnEJk'

    # TIFFs — organised by version, never deleted
    TIFF_DIR     = DATA_DIR

    # 🛑 FIX: Use explicit Path inside comprehension to avoid scoping errors
    TIFF_FILES   = [Path('/content/drive/MyDrive/Argus_Data') / f'Lahore_Daily_TimeSeries_{y}.tif' for y in range(2016, 2026)]

    TRAIN_START  = pd.Timestamp('2016-01-01')
    TRAIN_END    = pd.Timestamp('2026-02-01')

    # ── PERMANENT outputs (models, results, forecasts, optuna) ── NEVER DELETE ──
    PERM_DIR      = DATA_DIR / 'v7_0'
    MODELS_DIR    = PERM_DIR / 'models'
    RESULTS_DIR   = PERM_DIR / 'results'
    FORECASTS_DIR = PERM_DIR / 'forecasts'
    OPTUNA_DIR    = PERM_DIR / 'optuna'

    # ── TEMPORARY (caches + mid-training checkpoints) ── OK to delete if Drive full ──
    TEMP_DIR      = DATA_DIR / 'v7_0_temp'
    GRU_CKPT      = TEMP_DIR / 'gru_checkpoint_v7_0.pt'
    MLP_CKPT      = TEMP_DIR / 'mlp_checkpoint_v7_0.keras'
    DF_RAW_CACHE  = TEMP_DIR / 'df_raw_cache_v7_0.pkl'
    DF_FEAT_CACHE = TEMP_DIR / 'df_feat_cache_v7_0.pkl'
    WEATHER_CACHE = TEMP_DIR / 'weather_cache_v7_0.pkl'
    OPENAQ_CACHE  = TEMP_DIR / 'openaq_cache_v7_0.pkl'
    M2_CACHE      = TEMP_DIR / 'merra2_raw_cache_v7_0.pkl'

    BAND_INFO = {'modis_aod':0,'modis_aod055':1,'fmf':2,'uvai':3,
                 'no2':4,'so2':5,'co':6,'o3':7,'fire_india':8,'fire_pak':9,'ndvi':10}
    N_BANDS = 11

    GAS_FEATURES     = ['co','o3','no2','so2','uvai']
    WEATHER_FEATURES = ['wind_u','wind_v','relativehumidity_2m','temperature_2m',
                        'boundary_layer_height','cloudcover','surface_pressure',
                        'shortwave_radiation_sum','precipitation_sum','cloudcover_low']
    SMOG_MONTHS  = [10,11,12,1]
    TARGET_VARIABLE = 'pm2_5_aqi'
    AUX_TARGET   = 'aod'
    PATCH_HALF   = 5

    LAHORE_POINTS = [(31.5204,74.3587),(31.5000,74.2800),(31.5600,74.3900),
                     (31.5216,74.4036),(31.4700,74.4000)]
    LAT, LON = 31.5204, 74.3587
    INDIA_PUNJAB_BBOX = [74.0,29.5,77.5,32.5]
    PAK_PUNJAB_BBOX   = [70.0,29.5,74.0,33.0]

    VALIDATION_SPLIT = 0.20; TEST_SPLIT = 0.10; RANDOM_STATE = 42
    BATCH_SIZE=32; EPOCHS=150; LEARNING_RATE=1e-3; PATIENCE=20

    # [BUG-5 FIX] GRU_SEQ_LEN reduced 21->14; warmup added in Cell 15
    GRU_HIDDEN=128; GRU_LAYERS=2; GRU_SEQ_LEN=14; GRU_PATIENCE=15; GRU_EPOCHS=100

    OPTUNA_TRIALS=80; OPTUNA_CV_SPLITS=5
    MIN_CLEAR_DAYS=3; DAYS_BACK=35
    MAIAC_GEE_MAX_DAYS=60
    MERRA2_COLLECTION='NASA/GSFC/MERRA/flx/2'

    # [BUG-1 FIX] Explicit KEY_MAPs — no chained .replace() anywhere
    KEY_MAP_XGB = {
        'n':   'n_estimators',
        'lr':  'learning_rate',
        'd':   'max_depth',
        'ss':  'subsample',
        'cs':  'colsample_bytree',
        'mcw': 'min_child_weight',
        'ra':  'reg_alpha',
        'rl':  'reg_lambda',
    }
    KEY_MAP_LGB = {
        'n':   'n_estimators',
        'lr':  'learning_rate',
        'nl':  'num_leaves',
        'd':   'max_depth',
        'ss':  'subsample',
        'cs':  'colsample_bytree',
        'mcs': 'min_child_samples',
        'ra':  'reg_alpha',
        'rl':  'reg_lambda',
    }
    KEY_MAP_CAT = {
        'it':  'iterations',
        'lr':  'learning_rate',
        'd':   'depth',
        'l2':  'l2_leaf_reg',
        'ss':  'subsample',
        'cl':  'colsample_bylevel',
        'ml':  'min_data_in_leaf',
    }
    KEY_MAP_XGB_AOD = {
        'n':  'n_estimators',
        'lr': 'learning_rate',
        'd':  'max_depth',
        'ss': 'subsample',
        'cs': 'colsample_bytree',
    }
    KEY_MAP_LGB_AOD = {
        'n':  'n_estimators',
        'lr': 'learning_rate',
        'nl': 'num_leaves',
        'd':  'max_depth',
        'ss': 'subsample',
        'cs': 'colsample_bytree',
    }

    # NASA LANCE NRT config [BUG-4 FIX]
    LANCE_BASE  = 'https://nrt3.modaps.eosdis.nasa.gov/archive/allData/61/MCD19A2N/'
    LANCE_TILE  = 'h24v06'   # MODIS sinusoidal tile containing Lahore

config = Config()
def _remap(bp, keymap): return {keymap[k]: v for k, v in bp.items() if k in keymap}

for d in [config.PERM_DIR, config.MODELS_DIR, config.RESULTS_DIR,
          config.FORECASTS_DIR, config.OPTUNA_DIR, config.TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f'✅ Config v7.0')
print(f'   Permanent → {config.PERM_DIR}')
print(f'   Volatile  → {config.TEMP_DIR}  (safe to delete if Drive is full)')
print(f'   GRU_SEQ_LEN={config.GRU_SEQ_LEN} (was 21) | KEY_MAPs: explicit dicts')
# === Cell 3: Helpers ===
def pm25_to_aqi(pm25):
    bp=[(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
        (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)]
    if pd.isna(pm25) or pm25<0: return np.nan
    pm25=min(pm25,500.4)
    for c0,c1,i0,i1 in bp:
        if c0<=pm25<=c1: return round((i1-i0)/(c1-c0)*(pm25-c0)+i0)
    return 500

def aqi_to_category_us(aqi):
    if pd.isna(aqi): return('Unknown','#AAA')
    if aqi<=50:  return('Good','#00E400')
    if aqi<=100: return('Moderate','#FFFF00')
    if aqi<=150: return('Unhealthy for Sensitive Groups','#FF7E00')
    if aqi<=200: return('Unhealthy','#FF0000')
    if aqi<=300: return('Very Unhealthy','#8F3F97')
    return('Hazardous','#7E0023')

def paqm_to_category(pm25):
    if pd.isna(pm25) or pm25<0: return('Unknown','#AAA')
    if pm25<=35:  return('Good','#00C853')
    if pm25<=75:  return('Satisfactory','#FFD600')
    if pm25<=115: return('Moderately Polluted','#FF6D00')
    if pm25<=150: return('Poor','#DD2C00')
    if pm25<=250: return('Very Poor','#6A1B9A')
    return('Severe','#37474F')

def aqi_to_smog_level(aqi):
    if pd.isna(aqi): return('UNKNOWN','#AAA','No data')
    if aqi<=50:  return('HEALTHY','#00E400','No restriction needed')
    if aqi<=100: return('MODERATE','#FFFF00','Sensitive groups take care')
    if aqi<=150: return('UNHEALTHY','#FF7E00','Limit strenuous outdoor activity')
    if aqi<=200: return('VERY UNHEALTHY','#FF0000','Avoid outdoor; consider school closure')
    return        ('HAZARDOUS','#7E0023','Stay indoors — health emergency')

def aqi_to_cat_int(arr):
    out=[]
    for v in np.asarray(arr,dtype=float):
        if np.isnan(v): out.append(-1)
        elif v<=50:  out.append(0)
        elif v<=100: out.append(1)
        elif v<=150: out.append(2)
        elif v<=200: out.append(3)
        else:        out.append(4)
    return np.array(out)

def get_season(m):
    if m in[12,1,2]: return 'Winter'
    if m in[3,4,5]:  return 'Spring'
    if m in[6,7,8]:  return 'Summer'
    return 'Autumn'

def ckpt(obj, path, name):
    joblib.dump(obj, path); print(f'   💾 {name} → {path.name}')

def metrics(name, yt, yp):
    yt=np.asarray(yt,float); yp=np.asarray(yp,float)
    m=~(np.isnan(yt)|np.isnan(yp)); yt,yp=yt[m],yp[m]
    if len(yt)<2: return dict(Model=name,MAE=np.nan,RMSE=np.nan,R2=np.nan,Bias=np.nan,N=0)
    return dict(Model=name, MAE=mean_absolute_error(yt,yp),
                RMSE=np.sqrt(mean_squared_error(yt,yp)),
                R2=r2_score(yt,yp), Bias=float(np.mean(yp-yt)), N=int(m.sum()))

def _dagg(hourly, var, agg='mean'):
    daily=defaultdict(list)
    for i,ts in enumerate(hourly.get('time',[])):
        v=(hourly.get(var) or []); val=v[i] if i<len(v) else None
        if val is not None: daily[ts[:10]].append(float(val))
    if agg=='min':  return {d:float(np.min(vs))  for d,vs in daily.items()}
    if agg=='sum':  return {d:float(np.sum(vs))  for d,vs in daily.items()}
    return                  {d:float(np.mean(vs)) for d,vs in daily.items()}

PAQM=[(0,35,'#00C853',0.10,'Good'),(35,75,'#FFD600',0.09,'Satisfactory'),
      (75,115,'#FF6D00',0.09,'Mod.Poll.'),(115,150,'#DD2C00',0.09,'Poor'),
      (150,250,'#6A1B9A',0.07,'V.Poor'),(250,500,'#37474F',0.05,'Severe')]
print('✅ Helpers ready')
# === Cell 5: Weather + Real PM2.5 with API cache ===
def fetch_openaq_pm25(lat, lon, start_date, end_date):
    if config.OPENAQ_CACHE.exists():
        cached = joblib.load(config.OPENAQ_CACHE)
        print(f'   OpenAQ loaded from cache: {len(cached)} days')
        return cached
    pm25={}
    try:
        r=requests.get(f'https://api.openaq.org/v3/locations?coordinates={lat},{lon}'
                       f'&radius=30000&parameters_id=2&limit=5&order_by=lastUpdated&sort_order=desc',
                       timeout=15, headers={'X-API-Key':'demo'})
        locs=r.json().get('results',[])
        if not locs: print('   ⚠️  No OpenAQ sensors near Lahore'); return {}
        lid=locs[0]['id']; print(f'   OpenAQ station: {locs[0].get("name",lid)}')
        cur,end_ts=pd.Timestamp(start_date),pd.Timestamp(end_date)
        while cur<end_ts:
            chunk=min(cur+pd.Timedelta(days=89),end_ts)
            resp=requests.get(f'https://api.openaq.org/v3/locations/{lid}/measurements'
                              f'?date_from={cur.strftime("%Y-%m-%d")}&date_to={chunk.strftime("%Y-%m-%d")}'
                              f'&parameters_id=2&limit=10000',timeout=30,headers={'X-API-Key':'demo'})
            for row in resp.json().get('results',[]):
                try:
                    d=row['date']['local'][:10]; v=float(row['value'])
                    if v>=0: pm25.setdefault(d,[]).append(v)
                except: pass
            cur=chunk+pd.Timedelta(days=1)
        pm25={d:float(np.mean(vs)) for d,vs in pm25.items() if vs}
        print(f'   OpenAQ: {len(pm25)} real sensor days')
        joblib.dump(pm25, config.OPENAQ_CACHE)
    except Exception as e:
        print(f'   ⚠️  OpenAQ failed ({e}) — CAMS only')
    return pm25

def fetch_weather_and_pm25(lat, lon, start_date, end_date):
    cache_key = f'{start_date}_{end_date}'
    if config.WEATHER_CACHE.exists():
        cached = joblib.load(config.WEATHER_CACHE)
        if cached.get('key') == cache_key:
            print(f'   Weather loaded from cache')
            return cached['df']
    print(f'🌐 Weather+PM2.5: {start_date} → {end_date}')
    hvars=','.join(['windspeed_10m','winddirection_10m','relativehumidity_2m',
                    'temperature_2m','boundary_layer_height','cloudcover',
                    'surface_pressure','shortwave_radiation','precipitation','cloudcover_low'])
    url=(f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}'
         f'&hourly={hvars}&start_date={start_date}&end_date={end_date}&timezone=UTC')
    hourly=requests.get(url,timeout=90).json().get('hourly',{})
    wx=defaultdict(dict)
    for var in['windspeed_10m','winddirection_10m','relativehumidity_2m',
               'temperature_2m','cloudcover','surface_pressure','cloudcover_low']:
        for d,v in _dagg(hourly,var).items(): wx[d][var]=v
    for d,v in _dagg(hourly,'boundary_layer_height','mean').items(): wx[d]['boundary_layer_height']=v
    for d,v in _dagg(hourly,'boundary_layer_height','min').items():  wx[d]['blh_min']=v
    for d,v in _dagg(hourly,'shortwave_radiation','sum').items():    wx[d]['shortwave_radiation_sum']=v/1000.
    for d,v in _dagg(hourly,'precipitation','sum').items():          wx[d]['precipitation_sum']=v
    try:
        aq=requests.get(f'https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}'
                        f'&hourly=pm2_5&start_date={start_date}&end_date={end_date}&timezone=UTC',timeout=60).json()
        cams=_dagg(aq.get('hourly',{}),'pm2_5')
    except: cams={}
    openaq=fetch_openaq_pm25(lat,lon,start_date,end_date)
    rows=[]
    for d in sorted(set(wx)|set(cams)|set(openaq)):
        row={'date':pd.Timestamp(d)}; row.update(wx.get(d,{}))
        if d in openaq:
            row['pm2_5']=openaq[d]; row['pm2_5_source']='openaq_sensor'
        elif d in cams:
            row['pm2_5']=cams[d];   row['pm2_5_source']='cams_model'
        rows.append(row)
    df=pd.DataFrame(rows)
    if 'windspeed_10m' in df and 'winddirection_10m' in df:
        df['wind_u']=df['windspeed_10m']*np.cos(np.radians(df['winddirection_10m']))
        df['wind_v']=df['windspeed_10m']*np.sin(np.radians(df['winddirection_10m']))
    if 'pm2_5' in df:
        df['pm2_5_aqi'] =df['pm2_5'].apply(pm25_to_aqi)
        df['pm2_5_real']=(df.get('pm2_5_source','')=='openaq_sensor').astype(float)
        print(f'   PM2.5: {df["pm2_5"].notna().sum()} days | Real: {df["pm2_5_real"].sum():.0f} | CAMS: {(df["pm2_5"].notna().sum()-df["pm2_5_real"].sum()):.0f}')
    for pv in ['cloudcover','cloudcover_low']:
        if pv in df: df[pv]=df[pv].clip(0,100)/100.
    joblib.dump({'key':cache_key,'df':df}, config.WEATHER_CACHE)
    return df

# === Cell 7: Feature Engineering v7.0 ===
# [BUG-1 FIX] year_norm absent (confirmed by v6.0 SHAP — FIX-A working)
# NEW: aod_std_30, aod_trend_7/14, wind_india_3d/7d, monsoon_flag, heat_wave, dust_flag

def create_features_v7(df):
    df=df.copy().sort_values('date').reset_index(drop=True); eps=1e-10
    # Temporal
    df['month_sin']=np.sin(2*np.pi*df['month']/12)
    df['month_cos']=np.cos(2*np.pi*df['month']/12)
    df['doy_sin']=np.sin(2*np.pi*df['day_of_year']/365)
    df['doy_cos']=np.cos(2*np.pi*df['day_of_year']/365)
    df['smog_season_flag']=df['month'].isin(config.SMOG_MONTHS).astype(int)
    # year_norm deliberately absent
    df=pd.concat([df,pd.get_dummies(df['season'],prefix='season')],axis=1)

    # AOD lags
    for lag in [1,2,3,7,14,30]: df[f'aod_lag_{lag}']=df['aod'].shift(lag)
    df['aod_rolling_7'] =df['aod'].shift(1).rolling(7, min_periods=1).mean()
    df['aod_rolling_14']=df['aod'].shift(1).rolling(14,min_periods=1).mean()
    df['aod_rolling_30']=df['aod'].shift(1).rolling(30,min_periods=1).mean()
    df['aod_std_7']     =df['aod'].shift(1).rolling(7, min_periods=2).std().fillna(0)
    # [NEW-1] 30-day AOD std — captures sustained episode vs spike
    df['aod_std_30']    =df['aod'].shift(1).rolling(30,min_periods=10).std().fillna(0)
    df['aod_trend_3']   =df['aod_lag_1']-df['aod'].shift(3)
    # [NEW-2] 7-day and 14-day AOD trend — episode direction
    df['aod_trend_7']   =df['aod_lag_1']-df['aod_lag_7']
    df['aod_trend_14']  =df['aod_lag_1']-df['aod_lag_14']
    df['aod_clim_365']  =df['aod'].shift(1).rolling(365,min_periods=90).mean()
    df['aod_anomaly']   =df['aod_lag_1']-df['aod_clim_365']
    df['aod_imputed_lag1']=df['aod_imputed_flag'].shift(1).fillna(0)

    # Aerosol type
    if 'modis_aod055' in df:
        df['angstrom_exp']=np.clip(-np.log((df['modis_aod']+eps)/(df['modis_aod055']+eps))/np.log(0.47/0.55),-1,4)
        df['angstrom_exp_lag1']=df['angstrom_exp'].shift(1)
    if 'fmf' in df:
        df['fmf_lag1']=df['fmf'].shift(1)
        df['faod']=df['modis_aod']*df['fmf']; df['faod_lag1']=df['faod'].shift(1)

    # Gas lags
    for g in config.GAS_FEATURES:
        if g in df: df[f'{g}_lag1']=df[g].shift(1)
    if 'no2_lag1' in df and 'so2_lag1' in df: df['no2_so2_ratio']=df['no2_lag1']/(df['so2_lag1'].abs()+eps)
    if 'co_lag1' in df and 'no2_lag1' in df:  df['co_no2_ratio'] =df['co_lag1'] /(df['no2_lag1'].abs()+eps)

    # Fire features
    for fc in ['fire_india','fire_pak']:
        if fc in df:
            for s,f in [(1,'lag1'),(3,'rolling3'),(7,'rolling7'),(14,'rolling14')]:
                df[f'{fc}_{f}']=df[fc].shift(1).rolling(s,min_periods=1).sum() if 'rolling' in f else df[fc].shift(s)
    if 'fire_india_lag1' in df and 'fire_pak_lag1' in df:
        df['fire_total_lag1']    =df['fire_india_lag1']+df['fire_pak_lag1']
        df['fire_total_rolling7']=df['fire_india_rolling7']+df['fire_pak_rolling7']

    # NDVI
    if 'ndvi' in df: df['ndvi_lag16']=df['ndvi'].shift(16); df['ndvi_drop']=df['ndvi'].shift(16)-df['ndvi'].shift(32)

    # Weather-derived
    if 'surface_pressure' in df:
        df['pressure_anomaly']=df['surface_pressure']-df['surface_pressure'].rolling(30,min_periods=7).mean()
    if 'blh_min' in df:
        df['blh_min_lag1']=df['blh_min'].shift(1)
        df['inversion_flag']=(df['blh_min']<200).astype(float)
        df['inversion_flag_lag1']=df['inversion_flag'].shift(1)
    if 'relativehumidity_2m' in df and 'temperature_2m' in df:
        df['dewpoint_depression']=df['temperature_2m']-(df['temperature_2m']-((100-df['relativehumidity_2m'])/5.))
        df['fog_flag']=((df['relativehumidity_2m']>90)&(df['dewpoint_depression']<2.)&(df['smog_season_flag']==1)).astype(float)
        df['fog_flag_lag1']=df['fog_flag'].shift(1)
    if 'winddirection_10m' in df:
        df['wind_from_india']=((df['winddirection_10m']>=40)&(df['winddirection_10m']<=120)).astype(float)
        df['wind_from_desert']=((df['winddirection_10m']>=200)&(df['winddirection_10m']<=320)).astype(float)
        df['wind_from_india_lag1']=df['wind_from_india'].shift(1)
        df['wind_from_desert_lag1']=df['wind_from_desert'].shift(1)
        # [NEW-3] Wind persistence — sustained Indian transport episodes
        df['wind_india_3d']=df['wind_from_india'].shift(1).rolling(3,min_periods=1).mean()
        df['wind_india_7d']=df['wind_from_india'].shift(1).rolling(7,min_periods=1).mean()
    if 'shortwave_radiation_sum' in df and 'so2_lag1' in df: df['radiation_x_so2']=df['shortwave_radiation_sum']*df['so2_lag1']
    if 'precipitation_sum' in df:
        df['precip_lag1']=df['precipitation_sum'].shift(1)
        df['precip_rolling3']=df['precipitation_sum'].shift(1).rolling(3,min_periods=1).sum()
        # [NEW-4] Monsoon onset flag — biggest off-season regime signal
        rain_3d=df['precipitation_sum'].shift(1).rolling(3,min_periods=1).sum()
        df['monsoon_flag']=((rain_3d>15)&(df['month'].isin([6,7,8,9]))).astype(float)

    # [NEW-5] Heat wave flag — drives May-June photochemical smog
    if 'temperature_2m' in df:
        df['heat_wave']=(df['temperature_2m']>42).astype(float)
        df['heat_wave_3d']=df['temperature_2m'].shift(1).rolling(3,min_periods=1).mean()-40

    # [NEW-6] Dust flag — April-May Thar Desert transport
    if 'angstrom_exp_lag1' in df:
        ae=df['angstrom_exp'].fillna(1.0)
        fire_low=df.get('fire_total_lag1',pd.Series(0,index=df.index))<5
        df['dust_flag']=((ae<0.5)&fire_low&(df['month'].isin([3,4,5]))).astype(float)

    # Interactions
    if 'boundary_layer_height' in df: df['aod_x_blh']=df['aod_lag_1']*df['boundary_layer_height']
    if 'relativehumidity_2m' in df:   df['aod_x_humidity']=df['aod_lag_1']*df['relativehumidity_2m']
    df['aod_x_smog']=df['aod_lag_1']*df['smog_season_flag']

    # AQI lags + regime features
    if 'pm2_5' in df:
        real_aqi=df['pm2_5'].apply(pm25_to_aqi)
        df['aqi_lag_1']=real_aqi.shift(1)
        df['aqi_lag_7']=real_aqi.shift(7)
        df['aqi_rolling_7']=real_aqi.shift(1).rolling(7,min_periods=2).mean()
        df['aqi_pct_30d']=real_aqi.shift(1).rolling(30,min_periods=10).apply(
            lambda x: float(np.mean(x<=x.iloc[-1])) if len(x)>0 else np.nan)

    new_feats=['aod_std_30','aod_trend_7','aod_trend_14',
               'wind_india_3d','wind_india_7d','monsoon_flag','heat_wave','heat_wave_3d','dust_flag']
    lg=[f'{g}_lag1' for g in config.GAS_FEATURES if f'{g}_lag1' in df]
    tc=(['month_sin','month_cos','doy_sin','doy_cos','smog_season_flag',
         'aod_lag_1','aod_lag_2','aod_lag_3','aod_lag_7','aod_lag_14','aod_lag_30',
         'aod_rolling_7','aod_rolling_14','aod_rolling_30',
         'aod_std_7','aod_std_30','aod_trend_3','aod_trend_7','aod_trend_14',
         'aod_anomaly','aod_imputed_lag1']
        +[c for c in['angstrom_exp_lag1','fmf_lag1','faod_lag1'] if c in df]
        +[c for c in df if c.startswith('fire_') and any(c.endswith(s) for s in('lag1','rolling3','rolling7','rolling14'))]
        +[c for c in['ndvi_lag16','ndvi_drop'] if c in df]
        +[c for c in['aod_x_blh','aod_x_humidity','aod_x_smog'] if c in df]
        +[c for c in['no2_so2_ratio','co_no2_ratio'] if c in df]
        +[c for c in['aqi_lag_1','aqi_lag_7','aqi_rolling_7','aqi_pct_30d'] if c in df]
        +[c for c in new_feats if c in df]
        +[c for c in df if c.startswith('season_')])
    wd=[c for c in['pressure_anomaly','blh_min_lag1','inversion_flag','fog_flag_lag1',
                    'wind_from_india_lag1','wind_from_desert_lag1',
                    'radiation_x_so2','precip_lag1','precip_rolling3'] if c in df]
    return df,tc,lg,wd


# === Cell 20: Live Satellite Fetchers v7.0 ===
# [BUG-4 FIX] NASA LANCE NRT MAIAC primary (3-6hr latency), GEE fallback

def fetch_maiac_lance_nrt(lat, lon, days_back=5):
    """
    NASA LANCE NRT MAIAC — 3-6 hour latency.
    Requires NASA_APP_KEY in config for authorization.
    h24v06 tile covers Lahore. Downloads HDF4 granule, extracts AOD patch.
    """
    try:
        from pyhdf.SD import SD, SDC
    except ImportError:
        print('   pyhdf not installed; skipping LANCE NRT')
        return None, None

    import math, tempfile

    # Setup the authorization header
    headers = {'Authorization': f'Bearer {config.NASA_APP_KEY}'} if hasattr(config, 'NASA_APP_KEY') else {}
    if not headers:
        print('   ⚠️ NASA_APP_KEY not found in config. Download will likely return an HTML login page.')

    for db in range(0, days_back):
        d = (datetime.utcnow() - timedelta(days=db)).date()
        doy = d.timetuple().tm_yday
        url = f'{config.LANCE_BASE}{d.year}/{doy:03d}/'
        try:
            # 🛑 FIX: Added headers here
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200: continue
            tiles = re.findall(
                rf'MCD19A2N\.A\d+\.{config.LANCE_TILE}\.\S+?\.hdf', r.text)
            if not tiles:
                if db <= 2: print(f'   LANCE NRT: no {config.LANCE_TILE} tile for {d}')
                continue
            hdf_url = url + tiles[0]
            hdf_path = f'/tmp/maiac_lance_{d}.hdf'
            print(f'   Downloading LANCE NRT: {tiles[0]}...')

            # 🛑 FIX: Added headers here too, plus HTML fallback check
            with requests.get(hdf_url, headers=headers, stream=True, timeout=180) as resp:
                resp.raise_for_status()
                # Ensure we aren't saving HTML
                if 'text/html' in resp.headers.get('Content-Type', ''):
                    print(f'   ❌ LANCE NRT {d}: Received HTML instead of data. Check your NASA_APP_KEY.')
                    continue

                with open(hdf_path, 'wb') as f:
                    for chunk in resp.iter_content(65536): f.write(chunk)

            hdf = SD(hdf_path, SDC.READ)
            aod_raw = hdf.select('Optical_Depth_047').get().astype(float)
            # Try to also get AOD_055 and FMF if available
            try: aod055_raw = hdf.select('Optical_Depth_055').get().astype(float)
            except: aod055_raw = aod_raw.copy()
            try: fmf_raw = hdf.select('FineModeFraction_047').get().astype(float)
            except: fmf_raw = None
            hdf.end()
            import os; os.remove(hdf_path)

            # MODIS Sinusoidal projection: lat/lon → pixel row/col in h24v06 tile
            R = 6371007.181; T = 1111950.519; PIXELS = 1200
            lon_rad = math.radians(lon); lat_rad = math.radians(lat)
            sin_x = R * lon_rad * math.cos(lat_rad)
            sin_y = R * lat_rad
            tile_x0 = (24 - 18) * T
            tile_y0 = (9 - 6)  * T
            pixel_size = T / PIXELS
            col = int((sin_x - tile_x0) / pixel_size)
            row = int((tile_y0 - sin_y) / pixel_size)
            half = 5
            row = max(half, min(PIXELS - half - 1, row))
            col = max(half, min(PIXELS - half - 1, col))

            patch = aod_raw[row-half:row+half, col-half:col+half]
            valid = patch[(patch > 0) & (patch < 5000)]
            if valid.size < 4:
                print(f'   LANCE NRT {d}: patch has < 4 valid pixels')
                continue
            aod = float(valid.mean()) * 0.001

            patch055 = aod055_raw[row-half:row+half, col-half:col+half]
            v055 = patch055[(patch055 > 0) & (patch055 < 5000)]
            aod055 = float(v055.mean()) * 0.001 if v055.size >= 4 else aod

            fmf_val = 0.5
            if fmf_raw is not None:
                pfmf = fmf_raw[row-half:row+half, col-half:col+half]
                vfmf = pfmf[(pfmf >= 0) & (pfmf <= 1)]
                if vfmf.size >= 4: fmf_val = float(vfmf.mean())

            eps = 1e-10
            ae = float(np.clip(-math.log((aod+eps)/(aod055+eps))/math.log(0.47/0.55),-1,4))
            print(f'✅ LANCE NRT MAIAC: {d} ({db}d ago)  AOD={aod:.3f}  AE={ae:.2f}  FMF={fmf_val:.2f}')
            return d, {'modis_aod':aod,'modis_aod055':aod055,'fmf':fmf_val,
                       'angstrom_exp':ae,'faod':aod*fmf_val,'aod_source':'LANCE-NRT'}
        except Exception as e:
            print(f'   LANCE NRT {d}: {e}')
            continue
    return None, None

def find_latest_maiac_gee(max_lb=60):
    """GEE MAIAC fallback — searches backward up to max_lb days."""
    region=ee.Geometry.Point([config.LON,config.LAT]).buffer(20000)
    for db in range(1, max_lb+1):
        d=datetime.utcnow().date()-timedelta(days=db)
        ds=d.strftime('%Y-%m-%d'); dp=(d-timedelta(1)).strftime('%Y-%m-%d'); dn=(d+timedelta(1)).strftime('%Y-%m-%d')
        try:
            col=(ee.ImageCollection('MODIS/061/MCD19A2_GRANULES').filterDate(dp,dn)
                 .select(['Optical_Depth_047','Optical_Depth_055','FineModeFraction_047'])
                 .map(lambda i:i.updateMask(i.select('Optical_Depth_047').gt(0))))
            if col.size().getInfo()==0:
                if db<=10: print(f'  GEE empty {ds} (day -{db})...')
                continue
            res=col.median().reduceRegion(ee.Reducer.median(),region,scale=1000).getInfo()
            raw=(res or {}).get('Optical_Depth_047')
            if raw is None: continue
            aod=raw*0.001
            if not(0.01<=aod<=5.): continue
            raw055=(res or {}).get('Optical_Depth_055',raw); aod055=(raw055 or raw)*0.001
            fmf=float(np.clip((res or {}).get('FineModeFraction_047') or 0.5,0,1))
            eps=1e-10; ae=float(np.clip(-np.log((aod+eps)/(aod055+eps))/np.log(0.47/0.55),-1,4))
            print(f'✅ GEE MAIAC: {ds} ({db}d ago)  AOD={aod:.3f} AE={ae:.2f}')
            return d,{'modis_aod':aod,'modis_aod055':aod055,'fmf':fmf,'angstrom_exp':ae,'faod':aod*fmf,'aod_source':'MAIAC-GEE'}
        except Exception as e:
            if db<=5: print(f'  GEE err {ds}: {e}')
    print('⚠️  No MAIAC in GEE within lookback window'); return None,None

def fetch_gases(date_str, region):
    d=pd.Timestamp(date_str); ds=(d-pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    gases={}
    for col,band,key,def_ in[
        ('COPERNICUS/S5P/OFFL/L3_AER_AI','absorbing_aerosol_index','uvai',0.5),
        ('COPERNICUS/S5P/OFFL/L3_NO2','tropospheric_NO2_column_number_density','no2',2e-4),
        ('COPERNICUS/S5P/OFFL/L3_SO2','SO2_column_number_density','so2',1e-4),
        ('COPERNICUS/S5P/OFFL/L3_CO','CO_column_number_density','co',0.03),
        ('COPERNICUS/S5P/OFFL/L3_O3','O3_column_number_density','o3',0.13),
    ]:
        try:
            v=(ee.ImageCollection(col).filterDate(ds,date_str).select(band).mean()
               .reduceRegion(ee.Reducer.mean(),region,scale=2000).getInfo())
            gases[key]=float((v or {}).get(band,def_) or def_)
        except: gases[key]=def_
    return gases

def fetch_viirs_fire(ds):
    dn=(pd.Timestamp(ds)+pd.Timedelta(1,'d')).strftime('%Y-%m-%d'); out={'fire_india':0.,'fire_pak':0.}
    try:
        v=(ee.ImageCollection('FIRMS/VIIRS_SNPP_NRT').filterDate(ds,dn).select('T21').map(lambda i:i.gte(7)).sum())
        for k,bb in[('fire_india',config.INDIA_PUNJAB_BBOX),('fire_pak',config.PAK_PUNJAB_BBOX)]:
            cnt=v.reduceRegion(ee.Reducer.sum(),ee.Geometry.Rectangle(bb),scale=375).getInfo()
            out[k]=float((cnt or {}).get('T21',0) or 0)
    except: pass
    return out

def merra2_single(d):
    ds=d.strftime('%Y-%m-%d'); dn=(d+timedelta(1)).strftime('%Y-%m-%d')
    reg=ee.Geometry.Rectangle([config.LON-.5,config.LAT-.5,config.LON+.5,config.LAT+.5])
    try:
        col=ee.ImageCollection(config.MERRA2_COLLECTION).filterDate(ds,dn)
        res=col.select('TAUHGH').mean().add(col.select('TAULOW').mean()).reduceRegion(ee.Reducer.mean(),reg,50000).getInfo()
        val=list(res.values())[0] if res else None
        return float(val) if val else None
    except: return None

print('✅ v7.0 fetchers ready')
print('   Primary: NASA LANCE NRT MAIAC (3-6hr latency, h24v06 tile, HDF4 pixel extraction)')
print('   Fallback 1: GEE MAIAC (searches 60d back)')
print('   Fallback 2: MERRA-2 via GEE')
# === Cell 21: Weather Forecast Fetcher ===
def fetch_wx_forecast(lat, lon, days=2):
    hvars=','.join(['windspeed_10m','winddirection_10m','relativehumidity_2m',
                    'temperature_2m','boundary_layer_height','cloudcover',
                    'surface_pressure','shortwave_radiation','precipitation','cloudcover_low'])
    url=(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}'
         f'&hourly={hvars}&past_days=3&forecast_days={days}&timezone=Asia%2FKarachi')
    try: hourly=requests.get(url,timeout=30).json().get('hourly',{})
    except Exception as e: print(f'  ❌ WX forecast: {e}'); return {}
    res=defaultdict(dict)
    for var in['windspeed_10m','winddirection_10m','relativehumidity_2m','temperature_2m','cloudcover','surface_pressure','cloudcover_low']:
        for d,v in _dagg(hourly,var).items(): res[d][var]=v
    for d,v in _dagg(hourly,'boundary_layer_height','mean').items(): res[d]['boundary_layer_height']=v
    for d,v in _dagg(hourly,'boundary_layer_height','min').items():  res[d]['blh_min']=v
    for d,v in _dagg(hourly,'shortwave_radiation','sum').items():    res[d]['shortwave_radiation_sum']=v/1000.
    for d,v in _dagg(hourly,'precipitation','sum').items():          res[d]['precipitation_sum']=v
    out={}
    for d,vals in res.items():
        row=dict(vals)
        if 'windspeed_10m' in row and 'winddirection_10m' in row:
            row['wind_u']=row['windspeed_10m']*np.cos(np.radians(row['winddirection_10m']))
            row['wind_v']=row['windspeed_10m']*np.sin(np.radians(row['winddirection_10m']))
        for pv in['cloudcover','cloudcover_low']:
            if pv in row: row[pv]=min(row[pv],100)/100.
        out[d]=row
    return out
print('✅ fetch_wx_forecast ready')
# === RECOVERY CELL: Load ALL v7.0 Models from Drive ===
print("🔄 Restoring fully trained v7.0 models from Google Drive...")
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn

# --- Define the BiGRU Architecture so Python recognizes it ---
class BiGRU(nn.Module):
    def __init__(self, nf, h=128, nl=2, dp=0.25):
        super().__init__()
        self.gru = nn.GRU(nf, h, nl, batch_first=True, dropout=dp, bidirectional=True)
        self.attn = nn.Linear(h*2, 1)
        self.head = nn.Sequential(nn.Linear(h*2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        o, _ = self.gru(x)
        w = torch.softmax(self.attn(o), dim=1)
        return self.head((w*o).sum(1)).squeeze(-1)
# -------------------------------------------------------------

# 1. Load Scalers and Conformal Info
try:
    scaler_info = joblib.load(config.MODELS_DIR / 'scaler_info_v7_0.pkl')
    ci_info = joblib.load(config.MODELS_DIR / 'conformal_v7_0.pkl')
    W_GBM, W_GRU = ci_info['w_gbm'], ci_info['w_gru']
    print("  ✅ Scalers & Conformal weights loaded.")
except Exception as e:
    print(f"  ❌ Error loading scalers/CI: {e}")

# 2. Load Base GBM Trio
try:
    xgb_aqi_model = xgb.XGBRegressor()
    xgb_aqi_model.load_model(config.MODELS_DIR / 'xgb_aqi_v7_0.json')

    lgb_aqi_model = joblib.load(config.MODELS_DIR / 'lgb_aqi_v7_0.pkl')

    cat_aqi_model = CatBoostRegressor()
    cat_aqi_model.load_model(config.MODELS_DIR / 'cat_aqi_v7_0.cbm')
    print("  ✅ Base GBM Trio (XGB, LGB, CAT) loaded.")
except Exception as e:
    print(f"  ❌ Error loading Base GBM Trio: {e}")

# 3. Load Seasonal Models (Smog & Off-Season)
try:
    xgb_smog = xgb.XGBRegressor()
    xgb_smog.load_model(config.MODELS_DIR / 'xgb_smog_v7_0.json')

    lgb_off = joblib.load(config.MODELS_DIR / 'lgb_off_v7_0.pkl')
    print("  ✅ Seasonal Regime Models (Smog & Off-Season) loaded.")
except Exception as e:
    print(f"  ❌ Error loading Seasonal Models: {e}")

# 4. Load BiGRU
try:
    SL = config.GRU_SEQ_LEN
    gru = BiGRU(len(scaler_info['feature_names']), h=config.GRU_HIDDEN, nl=config.GRU_LAYERS)
    gru.load_state_dict(torch.load(config.MODELS_DIR / 'gru_best_v7_0.pt'))
    gru.eval()
    print("  ✅ BiGRU sequence model loaded.")
except Exception as e:
    print(f"  ❌ Error loading BiGRU: {e}")

print("\n🚀 Memory restored. Ready to run Live Prediction Pipeline!")
# === LIVE PREDICTION PIPELINE (T+0 and T+1) ===

# 1. Define the Weather Forecast Fetcher
def fetch_wx_forecast(lat, lon, days=2):
    hvars=','.join(['windspeed_10m','winddirection_10m','relativehumidity_2m',
                    'temperature_2m','boundary_layer_height','cloudcover',
                    'surface_pressure','shortwave_radiation','precipitation','cloudcover_low'])
    url=(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}'
         f'&hourly={hvars}&past_days=3&forecast_days={days}&timezone=Asia%2FKarachi')
    try: hourly=requests.get(url,timeout=30).json().get('hourly',{})
    except Exception as e: print(f'   ❌ WX forecast: {e}'); return {}
    res=defaultdict(dict)
    for var in['windspeed_10m','winddirection_10m','relativehumidity_2m','temperature_2m','cloudcover','surface_pressure','cloudcover_low']:
        for d,v in _dagg(hourly,var).items(): res[d][var]=v
    for d,v in _dagg(hourly,'boundary_layer_height','mean').items(): res[d]['boundary_layer_height']=v
    for d,v in _dagg(hourly,'boundary_layer_height','min').items():  res[d]['blh_min']=v
    for d,v in _dagg(hourly,'shortwave_radiation','sum').items():    res[d]['shortwave_radiation_sum']=v/1000.
    for d,v in _dagg(hourly,'precipitation','sum').items():          res[d]['precipitation_sum']=v
    out={}
    for d,vals in res.items():
        row=dict(vals)
        if 'windspeed_10m' in row and 'winddirection_10m' in row:
            row['wind_u']=row['windspeed_10m']*np.cos(np.radians(row['winddirection_10m']))
            row['wind_v']=row['windspeed_10m']*np.sin(np.radians(row['winddirection_10m']))
        for pv in['cloudcover','cloudcover_low']:
            if pv in row: row[pv]=min(row[pv],100)/100.
        out[d]=row
    return out

# 2. Run the Forecast Engine
print('🌍 v7.0 Autoregressive Forecast (T+0, T+1)...')
TODAY=datetime.utcnow().date()

print(f'\n📡 Step 1: Fetching Live MAIAC AOD...')
md, mdata = fetch_maiac_lance_nrt(config.LAT, config.LON, days_back=5)
if mdata is None:
    print('   LANCE NRT unavailable — trying GEE...')
    md, mdata = find_latest_maiac_gee(config.MAIAC_GEE_MAX_DAYS)
if mdata is None:
    print('   GEE MAIAC unavailable — trying MERRA-2...')
    ma = merra2_single(TODAY - timedelta(1))
    if ma:
        md = TODAY - timedelta(1)
        mdata = {'modis_aod':ma,'modis_aod055':ma,'fmf':0.5,'angstrom_exp':1.0,'faod':ma*0.5,'aod_source':'MERRA-2'}
        print(f'   MERRA-2 fallback: AOD={ma:.3f}')
    else:
        print('❌ No AOD data from any source. Cannot produce forecast.'); raise SystemExit

lag_days=(TODAY-md).days
print(f'   Source: {mdata["aod_source"]} | Date: {md} ({lag_days}d ago)')
print(f'   AOD={mdata["modis_aod"]:.3f}  AE={mdata["angstrom_exp"]:.2f}  FMF={mdata["fmf"]:.2f}')
if lag_days > 3: print(f'   ⚠️  AOD is {lag_days} days old')

print('\n📡 Step 2: Fetching Gas columns + fire data...')
region_c=ee.Geometry.Point([config.LON,config.LAT]).buffer(20000)
gas=fetch_gases(md.strftime('%Y-%m-%d'),region_c)
fire=fetch_viirs_fire(md.strftime('%Y-%m-%d'))
print(f'   NO2={gas.get("no2",0):.4f}  CO={gas.get("co",0):.4f}  UVAI={gas.get("uvai",0):.2f}')
print(f'   Fire India={fire["fire_india"]:.0f}  Fire Pak={fire["fire_pak"]:.0f}')

print('\n📊 Step 3: Building AOD history buffer...')
aod_buf={}
for db in range(1, config.DAYS_BACK+1):
    cd=TODAY-timedelta(days=db); cs=cd.strftime('%Y-%m-%d')
    dp=(cd-timedelta(1)).strftime('%Y-%m-%d'); dn=(cd+timedelta(1)).strftime('%Y-%m-%d')
    try:
        col=(ee.ImageCollection('MODIS/061/MCD19A2_GRANULES').filterDate(dp,dn)
             .select('Optical_Depth_047').map(lambda i:i.updateMask(i.gt(0))))
        if col.size().getInfo()==0: continue
        res=col.median().reduceRegion(ee.Reducer.median(),region_c,scale=1000).getInfo()
        raw=(res or {}).get('Optical_Depth_047')
        if raw is not None: aod_buf[cs]=raw*0.001
    except: pass
if mdata['aod_source']=='LANCE-NRT':
    aod_buf[md.strftime('%Y-%m-%d')]=mdata['modis_aod']
print(f'   Buffer: {len(aod_buf)} days loaded')

print('\n🌤️ Step 4: Fetching Weather forecast...')
wx=fetch_wx_forecast(config.LAT,config.LON,days=2)
hist_s=(TODAY-timedelta(config.DAYS_BACK)).strftime('%Y-%m-%d')
hist_e=(TODAY-timedelta(1)).strftime('%Y-%m-%d')
df_wxh=fetch_weather_and_pm25(config.LAT,config.LON,hist_s,hist_e)

print('\n⚙️ Step 5: Engineering features & scaling...')
hist_rows=[]
for ds,aod_v in sorted(aod_buf.items()):
    dts=pd.Timestamp(ds)
    row={'date':dts,'aod':aod_v,'modis_aod':aod_v,'modis_aod055':aod_v,'fmf':0.5,
         'uvai':0.,'fire_india':0.,'fire_pak':0.,'ndvi':np.nan,'aod_imputed_flag':0.,
         'year':dts.year,'month':dts.month,'day_of_year':dts.day_of_year,
         'season':get_season(dts.month),'smog_season':int(dts.month in config.SMOG_MONTHS)}
    wxr=df_wxh[df_wxh.date==dts]
    if not wxr.empty:
        for col in wxr.columns:
            if col!='date': row[col]=wxr[col].values[0]
    hist_rows.append(row)
ms=md.strftime('%Y-%m-%d')
for row in hist_rows:
    if row['date'].strftime('%Y-%m-%d')==ms:
        row.update(mdata); row.update({k:gas.get(k,0.) for k in['no2','so2','co','o3','uvai']}); row.update(fire)

fc_rows=[]
for fday,flabel in[(0,'T+0 (Today)'),(1,'T+1 (Tomorrow)')]:
    fd=TODAY+timedelta(fday); fds=fd.strftime('%Y-%m-%d')
    if fds not in wx: continue
    row={'date':pd.Timestamp(fd),'forecast_label':flabel,
         'year':fd.year,'month':fd.month,'day_of_year':fd.timetuple().tm_yday,
         'season':get_season(fd.month),'smog_season':int(fd.month in config.SMOG_MONTHS),
         'aod':np.nan,'modis_aod':mdata['modis_aod'],'modis_aod055':mdata['modis_aod055'],
         'fmf':mdata['fmf'],'uvai':gas.get('uvai',0.),'fire_india':fire['fire_india'],'fire_pak':fire['fire_pak'],
         'ndvi':np.nan,'aod_imputed_flag':float(lag_days>3),
         'no2':gas.get('no2',2e-4),'so2':gas.get('so2',1e-4),'co':gas.get('co',0.03),'o3':gas.get('o3',0.13)}
    row.update(wx[fds]); fc_rows.append(row)

df_comb=pd.DataFrame(hist_rows+fc_rows)
df_comb,_,_,_=create_features_v7(df_comb)
df_fc=df_comb.iloc[-len(fc_rows):].reset_index(drop=True)

Xfc=df_fc.reindex(columns=scaler_info['feature_names'],fill_value=0).ffill().bfill().fillna(0)
if scaler_info['robust_cols']:   Xfc[scaler_info['robust_cols']]  =scaler_info['robust'].transform(Xfc[scaler_info['robust_cols']])
if scaler_info['standard_cols']: Xfc[scaler_info['standard_cols']]=scaler_info['standard'].transform(Xfc[scaler_info['standard_cols']])

print('\n🧠 Step 6: Generating predictions...')
xf=np.clip(xgb_aqi_model.predict(Xfc.values),0,500)
lf=np.clip(lgb_aqi_model.predict(Xfc.values),0,500)
cf=np.clip(cat_aqi_model.predict(Xfc.values),0,500)
gbf=np.clip((xf+lf+cf)/3,0,500)

for i,(row,fday) in enumerate(zip(fc_rows,['T+0','T+1'])):
    month=pd.Timestamp(row['date']).month
    if month in config.SMOG_MONTHS and xgb_smog is not None:
        smog_pred=float(np.clip(xgb_smog.predict(Xfc.values[[i]]),0,500)[0])
        gbf[i]=0.5*gbf[i]+0.5*smog_pred
    elif month not in config.SMOG_MONTHS and lgb_off is not None:
        off_pred=float(np.clip(lgb_off.predict(Xfc.values[[i]]),0,500)[0])
        gbf[i]=0.5*gbf[i]+0.5*off_pred

if len(hist_rows)>=SL:
    Xha=pd.DataFrame(hist_rows+fc_rows); Xha,_,_,_=create_features_v7(Xha)
    Xha=Xha.reindex(columns=scaler_info['feature_names'],fill_value=0).ffill().bfill().fillna(0)
    if scaler_info['robust_cols']:   Xha[scaler_info['robust_cols']]  =scaler_info['robust'].transform(Xha[scaler_info['robust_cols']])
    if scaler_info['standard_cols']: Xha[scaler_info['standard_cols']]=scaler_info['standard'].transform(Xha[scaler_info['standard_cols']])
    Xsg=np.array([Xha.values[i-SL:i] for i in range(SL,len(Xha))],dtype=np.float32)
    gru.eval()
    with torch.no_grad(): grf=np.clip(gru(torch.tensor(Xsg)).numpy().flatten(),0,500)
    gf=grf[-len(fc_rows):]
else: gf=gbf.copy()
ensf=np.clip(W_GBM*gbf+W_GRU*gf,0,500)

fc_mo=[pd.Timestamp(r['date']).month for r in fc_rows]
qfc=[ci_info['q_hat_smog'] if m in config.SMOG_MONTHS else ci_info['q_hat_offseason'] for m in fc_mo]
cilo=np.clip(ensf-np.array(qfc),0,500); cihi=np.clip(ensf+np.array(qfc),0,500)

print('\n'+'═'*74)
print('           ARGUS v7.0 — LAHORE SMOG FORECAST')
print('═'*74)
print(f'  Generated : {TODAY}')
print(f'  Satellite : {mdata["aod_source"]} — {md} ({lag_days}d ago)')
print(f'  AOD={mdata["modis_aod"]:.3f}  AE={mdata["angstrom_exp"]:.2f}  FMF={mdata["fmf"]:.2f}  UVAI={gas.get("uvai",0):.2f}')
print(f'  Fire_India={fire["fire_india"]:.0f}  Fire_Pak={fire["fire_pak"]:.0f}')
print('─'*74)
for i,row in df_fc.iterrows():
    av=float(ensf[i]); lo=float(cilo[i]); hi=float(cihi[i])
    us,_=aqi_to_category_us(av); nq,_=paqm_to_category(av/4.)
    sl3,_,adv=aqi_to_smog_level(av)
    print(f'  {fc_rows[i]["forecast_label"]}  ({row["date"].date()})')
    print(f'    AQI (Ensemble)  : {av:.0f}   [90% CI: {lo:.0f} – {hi:.0f}]')
    print(f'    US EPA          : {us}')
    print(f'    Pakistan NEQS   : {nq}')
    print(f'    SMOG LEVEL      : {sl3}')
    print(f'    Advisory        : {adv}')
    print(f'    XGB={xf[i]:.0f}  LGB={lf[i]:.0f}  CAT={cf[i]:.0f}  GRU={gf[i]:.0f}  GBM Trio={gbf[i]:.0f}')
    print('─'*74)

# 7. Generate and save the forecast plot
flab=[r['forecast_label'] for r in fc_rows]
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,11))
fig.suptitle(f'ARGUS v7.0 — Lahore Forecast  ({TODAY})  [{mdata["aod_source"]}]', fontsize=14,fontweight='bold')

if aod_buf:
    ads=sorted(aod_buf.keys()); adv2=[aod_buf[d] for d in ads]
    clrs_aod=['#DB2877' if d==ms else '#1B3A6B' for d in ads]
    ax1.scatter(ads,adv2,c=clrs_aod,s=30,zorder=5)
    ax1.plot(ads,adv2,color='#1B3A6B',lw=1.5,alpha=0.6)
    ax1.axvline(ms,color='#DB2877',ls='--',lw=2, label=f'Latest MAIAC ({ms}, {mdata["aod_source"]})')
ax1.set_ylabel('AOD'); ax1.set_title(f'Satellite AOD History  (lag={lag_days}d)',fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(True,alpha=0.3); ax1.tick_params(axis='x',rotation=30)

for lo2,hi2,cl2,al2,_ in PAQM: ax2.axhspan(lo2,hi2,color=cl2,alpha=al2)
xfc=np.arange(len(flab))
bars=ax2.bar(xfc,ensf,color=['#EA580C','#2563EB'][:len(flab)], edgecolor='black',lw=1.5,width=0.5,alpha=0.88,zorder=5)
ax2.errorbar(xfc,ensf,yerr=[ensf-cilo,cihi-ensf],fmt='none', color='black',capsize=10,lw=2,zorder=6,label='90% CI')
for bar,av2 in zip(bars,ensf):
    sl4,_,_=aqi_to_smog_level(float(av2))
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+9, f'AQI {av2:.0f}\n{sl4}',ha='center',va='bottom',fontsize=12,fontweight='bold')
ax2.set_xticks(xfc); ax2.set_xticklabels(flab,fontsize=12)
ax2.axhline(150,color='red',lw=1.5,ls='--',alpha=0.8,label='AQI 150')
ax2.axhline(100,color='orange',lw=1.5,ls='--',alpha=0.8,label='AQI 100')
ax2.set_ylabel('US EPA AQI'); ax2.set_title('Forecast — Pakistan NEQS Colour Bands',fontweight='bold')
ax2.legend(loc='upper right',fontsize=9); ax2.grid(True,alpha=0.2)
ax2.set_ylim(0,max(500,float(np.max(cihi))*1.3))
plt.tight_layout()

fp=config.FORECASTS_DIR/f'forecast_{TODAY}_v7_0.png'
plt.savefig(fp,dpi=200,bbox_inches='tight'); plt.show()
print(f'\n✅ Forecast saved → {fp}')
print(f'   (Permanent record in {config.FORECASTS_DIR.name}/)')
