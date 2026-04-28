import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime

def get_season(m):
    if m in[12,1,2]: return 'Winter'
    if m in[3,4,5]:  return 'Spring'
    if m in[6,7,8]:  return 'Summer'
    return 'Autumn'

def create_features_v7(df):
    df=df.copy().sort_values('date').reset_index(drop=True); eps=1e-10
    cfg_smog_months = [10, 11, 12, 1]
    gas_features = ['co','o3','no2','so2','uvai']
    
    if 'month' not in df: df['month'] = [x.month for x in df['date']]
    if 'day_of_year' not in df: df['day_of_year'] = [x.timetuple().tm_yday for x in df['date']]
    if 'season' not in df: df['season'] = [get_season(m) for m in df['month']]

    # Temporal
    df['month_sin']=np.sin(2*np.pi*df['month']/12)
    df['month_cos']=np.cos(2*np.pi*df['month']/12)
    df['doy_sin']=np.sin(2*np.pi*df['day_of_year']/365)
    df['doy_cos']=np.cos(2*np.pi*df['day_of_year']/365)
    df['smog_season_flag']=df['month'].isin(cfg_smog_months).astype(int)
    # df=pd.concat([df,pd.get_dummies(df['season'],prefix='season')],axis=1)
    for s in ['Winter', 'Spring', 'Summer', 'Autumn']:
        df[f'season_{s}'] = (df['season'] == s).astype(int)

    # AOD lags
    for lag in [1,2,3,7,14,30]: df[f'aod_lag_{lag}']=df['aod'].shift(lag)
    df['aod_rolling_7'] =df['aod'].shift(1).rolling(7, min_periods=1).mean()
    df['aod_rolling_14']=df['aod'].shift(1).rolling(14,min_periods=1).mean()
    df['aod_rolling_30']=df['aod'].shift(1).rolling(30,min_periods=1).mean()
    df['aod_std_7']     =df['aod'].shift(1).rolling(7, min_periods=2).std().fillna(0)
    df['aod_std_30']    =df['aod'].shift(1).rolling(30,min_periods=10).std().fillna(0)
    df['aod_trend_3']   =df['aod_lag_1']-df['aod'].shift(3)
    df['aod_trend_7']   =df['aod_lag_1']-df['aod_lag_7']
    df['aod_trend_14']  =df['aod_lag_1']-df['aod_lag_14']
    df['aod_clim_365']  =df['aod'].shift(1).rolling(365,min_periods=90).mean()
    df['aod_anomaly']   =df['aod_lag_1']-df['aod_clim_365']
    if 'aod_imputed_flag' in df:
        df['aod_imputed_lag1']=df['aod_imputed_flag'].shift(1).fillna(0)
    else: df['aod_imputed_lag1'] = 0

    # Aerosol type
    if 'modis_aod055' in df and 'modis_aod' in df:
        df['angstrom_exp']=np.clip(-np.log((df['modis_aod']+eps)/(df['modis_aod055']+eps))/np.log(0.47/0.55),-1,4)
        df['angstrom_exp_lag1']=df['angstrom_exp'].shift(1)
    if 'fmf' in df and 'modis_aod' in df:
        df['fmf_lag1']=df['fmf'].shift(1)
        df['faod']=df['modis_aod']*df['fmf']; df['faod_lag1']=df['faod'].shift(1)

    # Gas lags
    for g in gas_features:
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
        df['wind_india_3d']=df['wind_from_india'].shift(1).rolling(3,min_periods=1).mean()
        df['wind_india_7d']=df['wind_from_india'].shift(1).rolling(7,min_periods=1).mean()
    if 'shortwave_radiation_sum' in df and 'so2_lag1' in df: df['radiation_x_so2']=df['shortwave_radiation_sum']*df['so2_lag1']
    if 'precipitation_sum' in df:
        df['precip_lag1']=df['precipitation_sum'].shift(1)
        df['precip_rolling3']=df['precipitation_sum'].shift(1).rolling(3,min_periods=1).sum()
        rain_3d=df['precipitation_sum'].shift(1).rolling(3,min_periods=1).sum()
        df['monsoon_flag']=((rain_3d>15)&(df['month'].isin([6,7,8,9]))).astype(float)

    if 'temperature_2m' in df:
        df['heat_wave']=(df['temperature_2m']>42).astype(float)
        df['heat_wave_3d']=df['temperature_2m'].shift(1).rolling(3,min_periods=1).mean()-40

    if 'angstrom_exp_lag1' in df and 'month' in df:
        ae=df['angstrom_exp'].fillna(1.0)
        fire_low=df.get('fire_total_lag1',pd.Series(0,index=df.index))<5
        df['dust_flag']=((ae<0.5)&fire_low&(df['month'].isin([3,4,5]))).astype(float)

    # Interactions
    if 'boundary_layer_height' in df: df['aod_x_blh']=df['aod_lag_1']*df['boundary_layer_height']
    if 'relativehumidity_2m' in df:   df['aod_x_humidity']=df['aod_lag_1']*df['relativehumidity_2m']
    df['aod_x_smog']=df['aod_lag_1']*df['smog_season_flag']

    # AQI lags
    if 'pm2_5_aqi' in df:
        df['aqi_lag_1']=df['pm2_5_aqi'].shift(1)
        df['aqi_lag_7']=df['pm2_5_aqi'].shift(7)
        df['aqi_rolling_7']=df['pm2_5_aqi'].shift(1).rolling(7,min_periods=2).mean()
        df['aqi_pct_30d']=df['pm2_5_aqi'].shift(1).rolling(30,min_periods=10).apply(
            lambda x: float(np.mean(x<=x.iloc[-1])) if len(x)>0 else np.nan)

    new_feats=['aod_std_30','aod_trend_7','aod_trend_14',
               'wind_india_3d','wind_india_7d','monsoon_flag','heat_wave','heat_wave_3d','dust_flag']
    lg=[f'{g}_lag1' for g in gas_features if f'{g}_lag1' in df]
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
    
    return df

class SmogDataPreprocessor:
    def __init__(self, scaler_path=None):
        if scaler_path is None:
            import os
            scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/scaler_info_v7_0.pkl"))
        
        self.scaler_path = scaler_path
        self.scaler_info = None
        self.load_artifacts()

    def load_artifacts(self):
        self.scaler_info = joblib.load(self.scaler_path)
        if not isinstance(self.scaler_info, dict) or 'feature_names' not in self.scaler_info:
            raise RuntimeError(
                f"scaler_info at {self.scaler_path} is malformed "
                f"(expected dict with 'feature_names')."
            )

    def preprocess_forecasts(self, hist_rows, fc_rows, SL):
        df_comb = pd.DataFrame(hist_rows + fc_rows)
        df_comb = create_features_v7(df_comb)
        
        # Get the feature names expected by the model
        feature_names = self.scaler_info['feature_names']
        
        # Prepare the dataframe for the last len(fc_rows) predictions
        df_fc = df_comb.iloc[-len(fc_rows):].reset_index(drop=True)
        Xfc = df_fc.reindex(columns=feature_names, fill_value=0).ffill().bfill().fillna(0)
        
        # Scale
        if self.scaler_info.get('robust_cols'):
            Xfc[self.scaler_info['robust_cols']] = self.scaler_info['robust'].transform(Xfc[self.scaler_info['robust_cols']])
        if self.scaler_info.get('standard_cols'):
            Xfc[self.scaler_info['standard_cols']] = self.scaler_info['standard'].transform(Xfc[self.scaler_info['standard_cols']])
            
        fc_scaled = Xfc.values
        
        # Now prepare the sliding windows for GRU (length SL)
        # We need historical sequences for T+0 and T+1
        if len(hist_rows) >= SL:
            Xha = df_comb.reindex(columns=feature_names, fill_value=0).ffill().bfill().fillna(0)
            if self.scaler_info.get('robust_cols'):
                Xha[self.scaler_info['robust_cols']] = self.scaler_info['robust'].transform(Xha[self.scaler_info['robust_cols']])
            if self.scaler_info.get('standard_cols'):
                Xha[self.scaler_info['standard_cols']] = self.scaler_info['standard'].transform(Xha[self.scaler_info['standard_cols']])
            
            Xsg = np.array([Xha.values[i-SL:i] for i in range(SL, len(Xha))], dtype=np.float32)
            # The last len(fc_rows) are the sequences to feed GRU
            seq_scaled = Xsg[-len(fc_rows):]
        else:
            seq_scaled = None
            
        return fc_scaled, seq_scaled


def aqi_to_category_us(aqi):
    if pd.isna(aqi): return('Unknown','#AAAAAA')
    if aqi<=50:  return('Good','#00E400')
    if aqi<=100: return('Moderate','#FFFF00')
    if aqi<=150: return('Unhealthy for Sensitive Groups','#FF7E00')
    if aqi<=200: return('Unhealthy','#FF0000')
    if aqi<=300: return('Very Unhealthy','#8F3F97')
    return('Hazardous','#7E0023')

def aqi_to_category(aqi_value):
    cat, color = aqi_to_category_us(aqi_value)
    return cat, color

def aqi_to_smog_level(aqi):
    """Returns (smog_level, color, advice, activity_ok_dict)."""
    if pd.isna(aqi):
        return ('UNKNOWN', '#AAA', 'No data',
                {'walking':False,'cycling':False,'school':False,'elderly':False,'asthmatic':False})
    if aqi <= 50:
        return ('HEALTHY', '#00E400', 'No restriction needed',
                {'walking':True,'cycling':True,'school':True,'elderly':True,'asthmatic':True})
    if aqi <= 100:
        return ('MODERATE', '#FFFF00', 'Sensitive groups take care',
                {'walking':True,'cycling':True,'school':True,'elderly':False,'asthmatic':False})
    if aqi <= 150:
        return ('UNHEALTHY', '#FF7E00', 'Limit strenuous outdoor activity',
                {'walking':True,'cycling':False,'school':True,'elderly':False,'asthmatic':False})
    if aqi <= 200:
        return ('VERY UNHEALTHY', '#FF0000', 'Avoid outdoor; consider school closure',
                {'walking':False,'cycling':False,'school':False,'elderly':False,'asthmatic':False})
    return ('HAZARDOUS', '#7E0023', 'Stay indoors - health emergency',
            {'walking':False,'cycling':False,'school':False,'elderly':False,'asthmatic':False})
