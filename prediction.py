import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostRegressor
from preprocessing import SmogDataPreprocessor, aqi_to_category, aqi_to_smog_level

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

class SmogPredictor:
    """
    Argus v7.0 Main Prediction Class
    """
    def __init__(self, models_dir=None):
        if models_dir is None:
            import os
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
        self.models_dir = models_dir
        self.preprocessor = SmogDataPreprocessor(f"{models_dir}/scaler_info_v7_0.pkl")
        self.load_status = {}
        self.SMOG_MONTHS = [10, 11, 12, 1]
        self.load_models()

    def _load_or_fail(self, name, loader_func):
        try:
            res = loader_func()
            self.load_status[name] = "Success"
            return res
        except Exception as e:
            self.load_status[name] = f"Failed: {str(e)}"
            return None

    def load_models(self):
        # Conformal & Weights
        def load_ci():
            # Uses joblib to load the conformal weights file
            return joblib.load(f"{self.models_dir}/conformal_v7_0.pkl")
            
        self.ci_info = self._load_or_fail("Conformal Weights", load_ci)

        if isinstance(self.ci_info, dict):
            self.W_GBM = self.ci_info.get('w_gbm', 0.8)
            self.W_GRU = self.ci_info.get('w_gru', 0.2)
        else:
            self.W_GBM, self.W_GRU = 0.8, 0.2

        # XGBoost
        def load_xgb():
            m = xgb.XGBRegressor()
            m.load_model(f"{self.models_dir}/xgb_aqi_v7_0.json")
            return m
        self.xgb_aqi = self._load_or_fail("XGBoost Base", load_xgb)

        # LightGBM
        def load_lgb():
            import lightgbm as lgb
            return joblib.load(f"{self.models_dir}/lgb_aqi_v7_0.pkl")
        self.lgb_aqi = self._load_or_fail("LightGBM Base", load_lgb)

        # CatBoost
        def load_cat():
            m = CatBoostRegressor()
            m.load_model(f"{self.models_dir}/cat_aqi_v7_0.cbm")
            return m
        self.cat_aqi = self._load_or_fail("CatBoost Base", load_cat)

        # XGB Smog
        def load_xgb_smog():
            m = xgb.XGBRegressor()
            m.load_model(f"{self.models_dir}/xgb_smog_v7_0.json")
            return m
        self.xgb_smog = self._load_or_fail("XGBoost Smog", load_xgb_smog)

        # LGB Off-season
        def load_lgb_off():
            import lightgbm as lgb
            return joblib.load(f"{self.models_dir}/lgb_off_v7_0.pkl")
        self.lgb_off = self._load_or_fail("LightGBM Off-Season", load_lgb_off)

        # BiGRU
        def load_gru():
            if not self.preprocessor.scaler_info: return None
            nf = len(self.preprocessor.scaler_info['feature_names'])
            m = BiGRU(nf, h=128, nl=2)
            m.load_state_dict(torch.load(f"{self.models_dir}/gru_best_v7_0.pt", weights_only=True))
            m.eval()
            return m
        self.gru = self._load_or_fail("BiGRU Seq", load_gru)

    def get_load_status(self):
        return self.load_status

    def predict_forecasts(self, hist_rows, fc_rows, SL=14):
        """
        Returns predictions for T+0 to T+3
        """
        fc_scaled, seq_scaled = self.preprocessor.preprocess_forecasts(hist_rows, fc_rows, SL)
        
        n_preds = len(fc_rows)
        gbf = np.zeros(n_preds)
        
        # 1. Base GBM Trio
        if self.xgb_aqi and self.lgb_aqi and self.cat_aqi:
            xf = np.clip(self.xgb_aqi.predict(fc_scaled), 0, 500)
            lf = np.clip(self.lgb_aqi.predict(fc_scaled), 0, 500)
            cf = np.clip(self.cat_aqi.predict(fc_scaled), 0, 500)
            gbf[:] = np.clip((xf + lf + cf) / 3, 0, 500)
        
        # 2. Add regime logic
        for i, row in enumerate(fc_rows):
            m = pd.Timestamp(row['date']).month
            if m in self.SMOG_MONTHS and self.xgb_smog:
                spred = float(np.clip(self.xgb_smog.predict(fc_scaled[[i]]), 0, 500)[0])
                gbf[i] = 0.5 * gbf[i] + 0.5 * spred
            elif m not in self.SMOG_MONTHS and self.lgb_off:
                opred = float(np.clip(self.lgb_off.predict(fc_scaled[[i]]), 0, 500)[0])
                gbf[i] = 0.5 * gbf[i] + 0.5 * opred

        # 3. Add GRU logic
        if self.gru and seq_scaled is not None:
            with torch.no_grad():
                grf = np.clip(self.gru(torch.tensor(seq_scaled)).numpy().flatten(), 0, 500)
            gf = grf
        else:
            gf = gbf.copy()

        # 4. Ensemble W
        ensf = np.clip(self.W_GBM * gbf + self.W_GRU * gf, 0, 500)

        # 5. CI
        fc_mo = [pd.Timestamp(r['date']).month for r in fc_rows]
        if self.ci_info:
            qfc = [self.ci_info['q_hat_smog'] if m in self.SMOG_MONTHS else self.ci_info['q_hat_offseason'] for m in fc_mo]
        else:
            qfc = [25.0] * n_preds
            
        cilo = np.clip(ensf - np.array(qfc), 0, 500)
        cihi = np.clip(ensf + np.array(qfc), 0, 500)

        results = []
        for i in range(n_preds):
            av = float(ensf[i])
            lo = float(cilo[i])
            hi = float(cihi[i])
            cat, color = aqi_to_category(av)
            sl, _, adv, _ = aqi_to_smog_level(av)
            
            results.append({
                'label': fc_rows[i].get('forecast_label', f'T+{i}'),
                'date': fc_rows[i]['date'].date(),
                'aqi': av,
                'ci_low': lo,
                'ci_high': hi,
                'aqi_category': cat,
                'aqi_color': color,
                'smog_level': sl,
                'recommendation': adv,
                'aod': fc_rows[i]['aod'] if not np.isnan(fc_rows[i]['aod']) else fc_rows[i]['modis_aod']
            })

        return results

    def get_recommendations(self, aqi, aod):
        pass # Replaced by aqi_to_smog_level

