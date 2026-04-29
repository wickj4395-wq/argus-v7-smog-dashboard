import ee
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import requests
import re
import math
from collections import defaultdict
import joblib
import os

def _dagg(hourly, var, agg='mean'):
    daily=defaultdict(list)
    for i,ts in enumerate(hourly.get('time',[])):
        v=(hourly.get(var) or []); val=v[i] if i<len(v) else None
        if val is not None: daily[ts[:10]].append(float(val))
    if agg=='min':  return {d:float(np.min(vs))  for d,vs in daily.items()}
    if agg=='sum':  return {d:float(np.sum(vs))  for d,vs in daily.items()}
    return {d:float(np.mean(vs)) for d,vs in daily.items()}

class EarthEngineDataFetcher:
    """
    Argus v7.0 Fetcher:
    Fetches real-time satellite data from NASA LANCE NRT, missing fallbacks from Google Earth Engine, 
    and historical weather from Open-Meteo.
    """
    
    def __init__(self, project_name="dazzling-trail-479218-p2", nasa_key=None, lat=31.5204, lon=74.3587):
        self.project_name = project_name
        self.nasa_key = nasa_key
        self.lat = lat
        self.lon = lon
        self.lance_base = 'https://nrt3.modaps.eosdis.nasa.gov/archive/allData/61/MCD19A2N/'
        self.tile_h, self.tile_v = self._modis_tile(self.lat, self.lon)
        self.lance_tile = f'h{self.tile_h:02d}v{self.tile_v:02d}'
        self.INDIA_PUNJAB_BBOX = [74.0,29.5,77.5,32.5]
        self.PAK_PUNJAB_BBOX   = [70.0,29.5,74.0,33.0]
        self.initialize_ee()

    @staticmethod
    def _modis_tile(lat, lon):
        R, T = 6371007.181, 1111950.519
        lon_rad, lat_rad = math.radians(lon), math.radians(lat)
        sin_x = R * lon_rad * math.cos(lat_rad)
        sin_y = R * lat_rad
        h = int(math.floor(sin_x / T)) + 18
        v = int(math.floor((9 * T - sin_y) / T))
        return h, v
    
    def initialize_ee(self):
        # Strategy 1: Service account from Streamlit secrets
        try:
            sa_info = dict(st.secrets["gcp_service_account"])
            # Fix common TOML issue: literal \\n in private_key must be real newlines
            if "private_key" in sa_info:
                sa_info["private_key"] = sa_info["private_key"].replace("\\n", "\n")
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/earthengine"]
            )
            ee.Initialize(credentials=credentials, project=self.project_name)
            return
        except Exception as e:
            st.warning(f"⚠️ Secrets auth failed: {type(e).__name__}: {e}")

        # Strategy 2: Service account JSON file on disk
        import json, glob
        for jf in sorted(glob.glob("*.json")):
            try:
                with open(jf) as f:
                    sa_info = json.load(f)
                if sa_info.get("type") == "service_account":
                    from google.oauth2 import service_account
                    credentials = service_account.Credentials.from_service_account_info(
                        sa_info,
                        scopes=["https://www.googleapis.com/auth/earthengine"]
                    )
                    ee.Initialize(credentials=credentials, project=self.project_name)
                    return
            except Exception:
                continue

        # Strategy 3: Default persistent credentials (local dev)
        try:
            ee.Initialize(project=self.project_name)
        except Exception as e:
            st.error(
                f"Earth Engine initialization failed: {str(e)}\n\n"
                "**Fix:** Add your service account JSON as `[gcp_service_account]` "
                "in Streamlit Cloud → Settings → Secrets."
            )
            raise

    def fetch_maiac_lance_nrt(self, days_back=7):
        """
        NASA LANCE NRT MAIAC — 3-6 hour latency.
        v8.0 fixes vs v7.0:
          FIX-A: wider patch half=12 (24x24 px) instead of 5 (10x10 px)
          FIX-B: AOD filter >= 0 instead of > 0 (includes valid clear-air pixels)
          FIX-C: verbose error logging replaces bare except:continue
        """
        try:
            from pyhdf.SD import SD, SDC
        except ImportError:
            print('[LANCE] pyhdf not installed -- skipping LANCE NRT')
            if self.nasa_key:
                st.warning('pyhdf not installed. LANCE NRT unavailable. Install via: pip install pyhdf')
            return None, None

        headers = {'Authorization': f'Bearer {self.nasa_key}'} if self.nasa_key else {}
        if not self.nasa_key:
            print('[LANCE] WARNING: no NASA_APP_KEY set. Directory listing may require auth.')

        for db in range(0, days_back):
            d = (datetime.utcnow() - timedelta(days=db)).date()
            doy = d.timetuple().tm_yday
            url = f'{self.lance_base}{d.year}/{doy:03d}/'
            print(f'[LANCE] Trying {d} (day -{db}) tile={self.lance_tile}')
            try:
                r = requests.get(url, headers=headers, timeout=15)
                print(f'[LANCE]   dir status: {r.status_code}')
                if r.status_code != 200:
                    continue
                tiles = re.findall(rf'MCD19A2N\.A\d+\.{self.lance_tile}\.\S+?\.hdf', r.text)
                print(f'[LANCE]   tiles found: {tiles}')
                if not tiles:
                    continue

                hdf_url = url + tiles[0]
                hdf_path = f'/tmp/maiac_lance_{d}.hdf'
                print(f'[LANCE]   downloading {tiles[0]}...')
                with requests.get(hdf_url, headers=headers, stream=True, timeout=180) as resp:
                    resp.raise_for_status()
                    ct = resp.headers.get('Content-Type', '')
                    if 'text/html' in ct:
                        print(f'[LANCE]   got HTML instead of HDF -- check NASA_APP_KEY')
                        continue
                    with open(hdf_path, 'wb') as f:
                        for chunk in resp.iter_content(65536): f.write(chunk)
                print(f'[LANCE]   file saved ({os.path.getsize(hdf_path)//1024} KB)')

                hdf = SD(hdf_path, SDC.READ)
                aod_raw = hdf.select('Optical_Depth_047').get().astype(float)
                print(f'[LANCE]   AOD array shape: {aod_raw.shape}')
                try: aod055_raw = hdf.select('Optical_Depth_055').get().astype(float)
                except Exception: aod055_raw = aod_raw.copy()
                try: fmf_raw = hdf.select('FineModeFraction_047').get().astype(float)
                except Exception: fmf_raw = None
                hdf.end()
                os.remove(hdf_path)

                R = 6371007.181; T = 1111950.519; PIXELS = 1200
                lon_rad = math.radians(self.lon); lat_rad = math.radians(self.lat)
                sin_x = R * lon_rad * math.cos(lat_rad)
                sin_y = R * lat_rad
                tile_x0 = (self.tile_h - 18) * T
                tile_y0 = (9 - self.tile_v) * T
                pixel_size = T / PIXELS
                col = int((sin_x - tile_x0) / pixel_size)
                row = int((tile_y0 - sin_y) / pixel_size)
                # FIX-A: wider patch — 24x24 px (~24 km) instead of 10x10 px
                half = 12
                row = max(half, min(PIXELS - half - 1, row))
                col = max(half, min(PIXELS - half - 1, col))
                print(f'[LANCE]   pixel row={row} col={col} patch={half*2}x{half*2}')

                patch = aod_raw[row-half:row+half, col-half:col+half]
                # FIX-B: include AOD=0 (clear air); fill value is -28672, not 0
                valid = patch[(patch >= 0) & (patch < 5000)]
                print(f'[LANCE]   valid pixels: {valid.size} of {patch.size}')
                if valid.size < 4:
                    print(f'[LANCE]   too few valid pixels, trying next day')
                    continue
                aod = float(valid.mean()) * 0.001

                patch055 = aod055_raw[row-half:row+half, col-half:col+half]
                v055 = patch055[(patch055 >= 0) & (patch055 < 5000)]
                aod055 = float(v055.mean()) * 0.001 if v055.size >= 4 else aod

                fmf_val = 0.5
                if fmf_raw is not None:
                    pfmf = fmf_raw[row-half:row+half, col-half:col+half]
                    vfmf = pfmf[(pfmf >= 0) & (pfmf <= 1)]
                    if vfmf.size >= 4: fmf_val = float(vfmf.mean())

                eps = 1e-10
                ae = float(np.clip(-math.log((aod+eps)/(aod055+eps))/math.log(0.47/0.55),-1,4))
                print(f'[LANCE] SUCCESS {d}: AOD={aod:.3f} AE={ae:.2f} FMF={fmf_val:.2f}')
                return d, {'modis_aod':aod,'modis_aod055':aod055,'fmf':fmf_val,
                           'angstrom_exp':ae,'faod':aod*fmf_val,'aod_source':'LANCE-NRT'}

            # FIX-C: log the actual error instead of silently continuing
            except Exception as e:
                print(f'[LANCE] day={d} EXCEPTION: {type(e).__name__}: {e}')
                continue

        print('[LANCE] All days exhausted -- returning None')
        return None, None

    def find_latest_maiac_gee(self, max_lb=60):
        region=ee.Geometry.Point([self.lon,self.lat]).buffer(20000)
        for db in range(1, max_lb+1):
            d=datetime.utcnow().date()-timedelta(days=db)
            dp=(d-timedelta(1)).strftime('%Y-%m-%d'); dn=(d+timedelta(1)).strftime('%Y-%m-%d')
            try:
                col=(ee.ImageCollection('MODIS/061/MCD19A2_GRANULES').filterDate(dp,dn)
                     .select(['Optical_Depth_047','Optical_Depth_055','FineModeFraction_047'])
                     .map(lambda i:i.updateMask(i.select('Optical_Depth_047').gt(0))))
                if col.size().getInfo()==0: continue
                res=col.median().reduceRegion(ee.Reducer.median(),region,scale=1000).getInfo()
                raw=(res or {}).get('Optical_Depth_047')
                if raw is None: continue
                aod=raw*0.001
                if not(0.01<=aod<=5.): continue
                raw055=(res or {}).get('Optical_Depth_055',raw); aod055=(raw055 or raw)*0.001
                fmf=float(np.clip((res or {}).get('FineModeFraction_047') or 0.5,0,1))
                eps=1e-10; ae=float(np.clip(-np.log((aod+eps)/(aod055+eps))/np.log(0.47/0.55),-1,4))
                return d,{'modis_aod':aod,'modis_aod055':aod055,'fmf':fmf,'angstrom_exp':ae,'faod':aod*fmf,'aod_source':'MAIAC-GEE'}
            except Exception:
                pass
        return None,None

    def merra2_single(self, d):
        ds=d.strftime('%Y-%m-%d'); dn=(d+timedelta(1)).strftime('%Y-%m-%d')
        reg=ee.Geometry.Rectangle([self.lon-.5,self.lat-.5,self.lon+.5,self.lat+.5])
        try:
            col=ee.ImageCollection('NASA/GSFC/MERRA/flx/2').filterDate(ds,dn)
            res=col.select('TAUHGH').mean().add(col.select('TAULOW').mean()).reduceRegion(ee.Reducer.mean(),reg,50000).getInfo()
            val=list(res.values())[0] if res else None
            return float(val) if val else None
        except: return None

    def fetch_gases(self, date_str):
        d=pd.Timestamp(date_str); ds=(d-pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        region=ee.Geometry.Point([self.lon,self.lat]).buffer(20000)
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

    def fetch_viirs_fire(self, ds):
        dn=(pd.Timestamp(ds)+pd.Timedelta(days=1)).strftime('%Y-%m-%d'); out={'fire_india':0.,'fire_pak':0.}
        try:
            v=(ee.ImageCollection('FIRMS/VIIRS_SNPP_NRT').filterDate(ds,dn).select('T21').map(lambda i:i.gte(7)).sum())
            for k,bb in[('fire_india',self.INDIA_PUNJAB_BBOX),('fire_pak',self.PAK_PUNJAB_BBOX)]:
                cnt=v.reduceRegion(ee.Reducer.sum(),ee.Geometry.Rectangle(bb),scale=375).getInfo()
                out[k]=float((cnt or {}).get('T21',0) or 0)
        except: pass
        return out

    def fetch_wx_forecast(self, days=4):
        hvars=','.join(['windspeed_10m','winddirection_10m','relativehumidity_2m',
                        'temperature_2m','boundary_layer_height','cloudcover',
                        'surface_pressure','shortwave_radiation','precipitation','cloudcover_low'])
        url=(f'https://api.open-meteo.com/v1/forecast?latitude={self.lat}&longitude={self.lon}'
             f'&hourly={hvars}&past_days=3&forecast_days={days}&timezone=Asia%2FKarachi')
        try: hourly=requests.get(url,timeout=30).json().get('hourly',{})
        except Exception: return {}
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

    def fetch_weather_and_pm25(self, start_date, end_date):
        hvars=','.join(['windspeed_10m','winddirection_10m','relativehumidity_2m',
                        'temperature_2m','boundary_layer_height','cloudcover',
                        'surface_pressure','shortwave_radiation','precipitation','cloudcover_low'])
        url=(f'https://archive-api.open-meteo.com/v1/archive?latitude={self.lat}&longitude={self.lon}'
             f'&hourly={hvars}&start_date={start_date}&end_date={end_date}&timezone=UTC')
        try: hourly=requests.get(url,timeout=90).json().get('hourly',{})
        except: return pd.DataFrame()
        wx=defaultdict(dict)
        for var in['windspeed_10m','winddirection_10m','relativehumidity_2m',
                   'temperature_2m','cloudcover','surface_pressure','cloudcover_low']:
            for d,v in _dagg(hourly,var).items(): wx[d][var]=v
        for d,v in _dagg(hourly,'boundary_layer_height','mean').items(): wx[d]['boundary_layer_height']=v
        for d,v in _dagg(hourly,'boundary_layer_height','min').items():  wx[d]['blh_min']=v
        for d,v in _dagg(hourly,'shortwave_radiation','sum').items():    wx[d]['shortwave_radiation_sum']=v/1000.
        for d,v in _dagg(hourly,'precipitation','sum').items():          wx[d]['precipitation_sum']=v
        try:
            aq=requests.get(f'https://air-quality-api.open-meteo.com/v1/air-quality?latitude={self.lat}&longitude={self.lon}'
                            f'&hourly=pm2_5&start_date={start_date}&end_date={end_date}&timezone=UTC',timeout=60).json()
            cams=_dagg(aq.get('hourly',{}),'pm2_5')
        except: cams={}
        
        # openaq fallback
        openaq = {}
        try:
            r=requests.get(f'https://api.openaq.org/v3/locations?coordinates={self.lat},{self.lon}&radius=30000&parameters_id=2&limit=5', timeout=15)
            locs=r.json().get('results',[])
            if locs:
                lid=locs[0]['id']
                resp=requests.get(f'https://api.openaq.org/v3/locations/{lid}/measurements?date_from={start_date}&date_to={end_date}&parameters_id=2&limit=10000',timeout=30)
                pm25 = defaultdict(list)
                for row in resp.json().get('results',[]):
                    d=row['date']['local'][:10]; v=float(row['value'])
                    if v>=0: pm25[d].append(v)
                openaq={d:np.mean(vs) for d,vs in pm25.items()}
        except: pass

        rows=[]
        for d in sorted(set(wx)|set(cams)|set(openaq)):
            row={'date':pd.Timestamp(d)}; row.update(wx.get(d,{}))
            if d in openaq: row['pm2_5']=openaq[d]; row['pm2_5_source']='openaq_sensor'
            elif d in cams: row['pm2_5']=cams[d];   row['pm2_5_source']='cams_model'
            rows.append(row)
        df=pd.DataFrame(rows)
        if 'windspeed_10m' in df and 'winddirection_10m' in df:
            df['wind_u']=df['windspeed_10m']*np.cos(np.radians(df['winddirection_10m']))
            df['wind_v']=df['windspeed_10m']*np.sin(np.radians(df['winddirection_10m']))
        def pm25_to_aqi(pm25):
            bp=[(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
                (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)]
            if pd.isna(pm25) or pm25<0: return np.nan
            pm25=min(pm25,500.4)
            for c0,c1,i0,i1 in bp:
                if c0<=pm25<=c1: return round((i1-i0)/(c1-c0)*(pm25-c0)+i0)
            return 500
        if 'pm2_5' in df:
            df['pm2_5_aqi'] =df['pm2_5'].apply(pm25_to_aqi)
        for pv in ['cloudcover','cloudcover_low']:
            if pv in df: df[pv]=df[pv].clip(0,100)/100.
        return df

    def openmeteo_aod_single(self, d):
        ds = d.strftime('%Y-%m-%d')
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={self.lat}&longitude={self.lon}&hourly=aerosol_optical_depth&start_date={ds}&end_date={ds}&timezone=UTC"
        try:
            r = requests.get(url, timeout=30).json()
            vals = [v for v in r.get('hourly', {}).get('aerosol_optical_depth', []) if v is not None]
            if vals:
                return float(np.mean(vals))
        except:
            pass
        return None

    def fetch_all_data(self):
        TODAY=datetime.utcnow().date()
        import concurrent.futures

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        future_wx = executor.submit(self.fetch_wx_forecast, 4)
        future_wxh = executor.submit(self.fetch_weather_and_pm25, (TODAY-timedelta(36)).strftime('%Y-%m-%d'), (TODAY-timedelta(1)).strftime('%Y-%m-%d'))

        # v8.0 corrected fallback order: LANCE-NRT -> MERRA-2 (~1d) -> MAIAC-GEE (last resort)
        md, mdata = self.fetch_maiac_lance_nrt(days_back=7)
        if not mdata:
            print('[Fetcher] LANCE unavailable -- trying MERRA-2 (~1d lag)...')
            ma = self.merra2_single(TODAY - timedelta(1))
            if ma:
                md = TODAY - timedelta(1)
                mdata = {'modis_aod':ma,'modis_aod055':ma,'fmf':0.5,'angstrom_exp':1.0,'faod':ma*0.5,'aod_source':'MERRA-2'}
                print(f'[Fetcher] MERRA-2 fallback: AOD={ma:.3f} (lag=1d)')
        if not mdata:
            print('[Fetcher] MERRA-2 unavailable -- trying MAIAC-GEE (40-60d lag, last resort)...')
            md, mdata = self.find_latest_maiac_gee(90)
        if not mdata:
                oma = self.openmeteo_aod_single(TODAY)
                if oma is not None:
                    md = TODAY
                    mdata = {'modis_aod':oma,'modis_aod055':oma,'fmf':0.5,'angstrom_exp':1.0,'faod':oma*0.5,'aod_source':'Open-Meteo CAMS'}
                else:
                    raise ValueError("No AOD data available from any source.")

        ms = md.strftime('%Y-%m-%d')
        future_gas = executor.submit(self.fetch_gases, ms)
        future_fire = executor.submit(self.fetch_viirs_fire, ms)
        
        # Buffer batched in EE
        def fetch_ee_buffer():
            buf={}
            region_c=ee.Geometry.Point([self.lon,self.lat]).buffer(20000)
            try:
                end_date_str = TODAY.strftime('%Y-%m-%d')
                days_list = ee.List.sequence(1, 35)
                def get_daily(db):
                    cd = ee.Date(end_date_str).advance(ee.Number(db).multiply(-1), 'day')
                    dp = cd.advance(-1, 'day')
                    dn = cd.advance(1, 'day')
                    daily_col = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES').filterDate(dp,dn).select('Optical_Depth_047').map(lambda i:i.updateMask(i.gt(0)))
                    med = daily_col.median()
                    res = med.reduceRegion(ee.Reducer.median(), region_c, scale=1000)
                    return ee.Feature(None, {'date': cd.format('YYYY-MM-dd'), 'val': res.get('Optical_Depth_047')})
                
                fc = ee.FeatureCollection(days_list.map(get_daily))
                info = fc.getInfo()
                for feat in info.get('features', []):
                    props = feat.get('properties', {})
                    val = props.get('val')
                    if val is not None:
                        buf[props['date']] = float(val) * 0.001
            except Exception as e:
                print(f"[EE Buffer] batch error: {e}")
                
            if md:
                buf[md.strftime('%Y-%m-%d')] = mdata['modis_aod']
            return buf

        future_buf = executor.submit(fetch_ee_buffer)

        gas = future_gas.result()
        fire = future_fire.result()
        aod_buf = future_buf.result()
        wx = future_wx.result()
        df_wxh = future_wxh.result()
        executor.shutdown()
        
        def get_season(m):
            if m in[12,1,2]: return 'Winter'
            if m in[3,4,5]:  return 'Spring'
            if m in[6,7,8]:  return 'Summer'
            return 'Autumn'

        hist_rows=[]
        for ds,aod_v in sorted(aod_buf.items()):
            dts=pd.Timestamp(ds)
            row={'date':dts,'aod':aod_v,'modis_aod':aod_v,'modis_aod055':aod_v,'fmf':0.5,
                 'uvai':0.,'fire_india':0.,'fire_pak':0.,'ndvi':np.nan,'aod_imputed_flag':0.,
                 'year':dts.year,'month':dts.month,'day_of_year':dts.day_of_year,
                 'season':get_season(dts.month),'smog_season':int(dts.month in [10,11,12,1])}
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
        for fday,flabel in[(0,'T+0 (Today)'),(1,'T+1 (Tomorrow)'),(2,'T+2 (Day After)'),(3,'T+3 (In 3 Days)')]:
            fd=TODAY+timedelta(fday); fds=fd.strftime('%Y-%m-%d')
            if fds not in wx: continue
            row={'date':pd.Timestamp(fd),'forecast_label':flabel,
                 'year':fd.year,'month':fd.month,'day_of_year':fd.timetuple().tm_yday,
                 'season':get_season(fd.month),'smog_season':int(fd.month in [10,11,12,1]),
                 'aod':np.nan,'modis_aod':mdata['modis_aod'],'modis_aod055':mdata['modis_aod055'],
                 'fmf':mdata['fmf'],'uvai':gas.get('uvai',0.),'fire_india':fire['fire_india'],'fire_pak':fire['fire_pak'],
                 'ndvi':np.nan,'aod_imputed_flag':float((TODAY-md).days>3),
                 'no2':gas.get('no2',2e-4),'so2':gas.get('so2',1e-4),'co':gas.get('co',0.03),'o3':gas.get('o3',0.13)}
            row.update(wx[fds]); fc_rows.append(row)

        return hist_rows, fc_rows, mdata, gas, fire

