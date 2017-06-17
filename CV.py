import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
import xgboost as xgb

from ZK import ZKTools

# --------------------
print('Loading data...')
path = '/Users/Bing/Documents/DS/Zillow_Kaggle/'
df_train = pd.read_csv('train_features.csv')
df_target = pd.read_csv('train_target.csv').values.ravel()

imp = Imputer()
df_train_imp = pd.DataFrame(imp.fit_transform(df_train), columns = df_train.columns)


n_estimators = 1
random_state = 0
# RF
rf_params = {
    'n_jobs': -1,
    'n_estimators': n_estimators,
    'warm_start': True,
    'max_depth' : 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
    }
rf = RandomForestRegressor(random_state = random_state, **rf_params)
rf_CV = ZKTools.CV(df_train = df_train_imp, df_target = df_target, n_splits = 10, model = rf, params = rf_params)
mae_mean, mae_std = rf_CV.cross_validate()
rf_CV.report('rf', '6_17_v1', mae_mean, mae_std)

xgb_params = {
    'n_estimators' : n_estimators,
    'learning_rate' : 0.02,
    'max_depth' : 6,
    'objective' : 'reg:linear',
    'silent' : True
    }
xgb = xgb.XGBRegressor(**xgb_params)
xgb_CV = ZKTools.CV(df_train = df_train_imp, df_target = df_target, n_splits = 10, model = xgb, params = xgb_params)
mae_mean, mae_std = xgb_CV.cross_validate()
xgb_CV.report('xgb', '6_17_v1', mae_mean, mae_std)

