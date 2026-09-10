import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor

# -------- File Paths --------
prediction_file = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Predictions\prediction_input_to use.xlsx'  # <- Your prediction input
classification_train_file = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Obj2_Classification input.xlsx'
regression_train_file = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Obj2_Regression input.xlsx'
output_file = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Predictions\multi_stage_predictions.xlsx'

# --------- Train Classifier ---------
df_class = pd.read_excel(classification_train_file).dropna(how='all')
X_class = df_class.drop(columns=['Slope state', 'Slope type', 'No of days for failure', 'PWP at failure'], errors='ignore')
y_class = df_class['Slope state']
cat_feat = ['Rainfall realisation']
for c in cat_feat:
    X_class[c] = X_class[c].astype('category')

params_lgbm = {
    'max_depth': 8, 'n_estimators': 500, 'learning_rate': 0.1,
    'num_leaves': 31, 'min_child_samples': 20, 'reg_alpha': 3,
    'reg_lambda': 1, 'colsample_bytree': 0.6, 'objective': 'binary',
    'verbose': -1
}
clf = lgb.LGBMClassifier(**params_lgbm)
clf.fit(X_class, y_class, categorical_feature=cat_feat)

# --------- Train Regression Models ---------
df_regr = pd.read_excel(regression_train_file).dropna(how='any')
X_regr = df_regr.drop(columns=['No of days for failure', 'PWP at failure'])
y_days = df_regr['No of days for failure']
y_pwp = df_regr['PWP at failure']
for c in cat_feat:
    X_regr[c] = X_regr[c].astype('category')

cb_params_no_days = {
    'depth': 8, 'learning_rate': 0.1, 'iterations': 100,
    'l2_leaf_reg': 3, 'border_count': 32, 'random_strength': 0,
    'loss_function': 'RMSE', 'silent': True, 'cat_features': cat_feat
}
catboost_days = CatBoostRegressor(**cb_params_no_days)
catboost_days.fit(X_regr, y_days)

lgbm_params_pwp = {
    'num_leaves': 31, 'learning_rate': 0.01, 'n_estimators': 200, 'min_child_samples': 10,
    'reg_alpha': 1, 'reg_lambda': 1, 'random_state': 42, 'verbose': -1
}
lgbm_pwp = lgb.LGBMRegressor(**lgbm_params_pwp)
lgbm_pwp.fit(X_regr, y_pwp, categorical_feature=cat_feat)

# --------- Prediction Stage ---------
df_pred = pd.read_excel(prediction_file)
for c in cat_feat:
    df_pred[c] = df_pred[c].astype('category')

# Predict Slope state
class_pred = clf.predict(df_pred)
df_pred['Slope state'] = class_pred

# Prepare outputs with default NaN
df_pred['No of days for failure'] = np.nan
df_pred['PWP at failure'] = np.nan

# Only predict regression for "Unstable"
mask_failure = (df_pred['Slope state'] == 'Unstable')
if mask_failure.any():
    df_failure = df_pred.loc[mask_failure].copy()
    # Use the *exact* input columns for regression as were used during model training!
    df_failure_regr = df_failure[X_regr.columns]
    df_pred.loc[mask_failure, 'No of days for failure'] = catboost_days.predict(df_failure_regr)
    df_pred.loc[mask_failure, 'PWP at failure'] = lgbm_pwp.predict(df_failure_regr)

# --------- Save Predictions ---------
df_pred.to_excel(output_file, index=False)
print(f"Multi-stage predictions saved to {output_file}")
