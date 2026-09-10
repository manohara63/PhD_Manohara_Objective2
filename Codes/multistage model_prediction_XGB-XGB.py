import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder

# -------- File Paths --------
prediction_file = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Predictions\prediction_input_to use.xlsx'
classification_train_file = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Classification input_refined.xlsx'
regression_train_file = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Regression input_refined.xlsx'
output_file = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Predictions\multi_stage_predictions_XGBoost_refined.xlsx'

# ============================================================
# 1. Train XGBoost Classifier (Slope state)
# ============================================================

df_class = pd.read_excel(classification_train_file).dropna(how='all')

X_class = df_class.drop(
    columns=['Slope state', 'Slope type', 'No of days for failure', 'PWP at failure'],
    errors='ignore'
)
y_class = df_class['Slope state']

# Encode labels: Stable/Unstable -> 0/1
label_encoder = LabelEncoder()
y_class_encoded = label_encoder.fit_transform(y_class)

# One-hot encode categorical for XGBoost
X_class_encoded = pd.get_dummies(X_class, columns=['Rainfall realisation'], drop_first=False)

# Best hyperparameters for classification
params_xgb_class = {
    'subsample': 1.0,
    'reg_lambda': 0.1,
    'reg_alpha': 0,
    'n_estimators': 200,
    'min_child_weight': 1,
    'max_depth': 5,
    'learning_rate': 0.2,
    'gamma': 0.3,
    'colsample_bytree': 1.0
}

clf = XGBClassifier(
    random_state=42,
    verbosity=0,
    objective='binary:logistic',
    eval_metric='logloss',
    **params_xgb_class
)
clf.fit(X_class_encoded, y_class_encoded)

# ============================================================
# 2. Train XGBoost Regression Models (Days & PWP)
# ============================================================

df_regr = pd.read_excel(regression_train_file).dropna(how='any')

X_regr = df_regr.drop(columns=['No of days for failure', 'PWP at failure'])
y_days = df_regr['No of days for failure']
y_pwp = df_regr['PWP at failure']

# One-hot encode for XGBoost regression
X_regr_encoded = pd.get_dummies(X_regr, columns=['Rainfall realisation'], drop_first=False)

# Best hyperparameters for regression – No of days for failure
params_xgb_days = {
    'max_depth': 7,
    'learning_rate': 0.01,
    'n_estimators': 1000,
    'min_child_weight': 1,
    'gamma': 0.2,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}

xgb_days = XGBRegressor(
    random_state=42,
    verbosity=0,
    objective='reg:squarederror',
    tree_method='hist',
    **params_xgb_days
)
xgb_days.fit(X_regr_encoded, y_days)

# Best hyperparameters for regression – PWP at failure
params_xgb_pwp = {
    'max_depth': 7,
    'learning_rate': 0.01,
    'n_estimators': 200,
    'min_child_weight': 3,
    'gamma': 0.2,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'reg_alpha': 0.1,
    'reg_lambda': 0
}

xgb_pwp = XGBRegressor(
    random_state=42,
    verbosity=0,
    objective='reg:squarederror',
    tree_method='hist',
    **params_xgb_pwp
)
xgb_pwp.fit(X_regr_encoded, y_pwp)

# ============================================================
# 3. Prediction Stage
# ============================================================

df_pred = pd.read_excel(prediction_file)

# One-hot encode prediction data
df_pred_encoded = pd.get_dummies(df_pred, columns=['Rainfall realisation'], drop_first=False)

# --- Align columns for classifier ---
missing_cols_class = set(X_class_encoded.columns) - set(df_pred_encoded.columns)
for col in missing_cols_class:
    df_pred_encoded[col] = 0
# Extra cols in prediction but not in training are ignored by ordering
df_pred_encoded = df_pred_encoded[X_class_encoded.columns]

# Predict Slope state
class_pred_numeric = clf.predict(df_pred_encoded)
class_pred = label_encoder.inverse_transform(class_pred_numeric)  # 0->Stable, 1->Unstable
df_pred['Slope state'] = class_pred

# Initialize regression outputs
df_pred['No of days for failure'] = np.nan
df_pred['PWP at failure'] = np.nan

# Only predict regression for "Unstable"
mask_failure = (df_pred['Slope state'] == 'Unstable')

if mask_failure.any():
    # Subset encoded prediction rows for unstable slopes
    df_failure_encoded = df_pred_encoded.loc[mask_failure].copy()

    # --- Align columns for regression models ---
    missing_cols_regr = set(X_regr_encoded.columns) - set(df_failure_encoded.columns)
    for col in missing_cols_regr:
        df_failure_encoded[col] = 0
    df_failure_encoded = df_failure_encoded[X_regr_encoded.columns]

    # Predict days to failure and PWP
    days_pred = xgb_days.predict(df_failure_encoded)
    pwp_pred = xgb_pwp.predict(df_failure_encoded)

    # (Optional) enforce non‑negative predictions
    days_pred = np.maximum(days_pred, 0)
    pwp_pred = np.maximum(pwp_pred, 0)

    df_pred.loc[mask_failure, 'No of days for failure'] = days_pred
    df_pred.loc[mask_failure, 'PWP at failure'] = pwp_pred

# ============================================================
# 4. Save Predictions
# ============================================================

df_pred.to_excel(output_file, index=False)
print(f"XGBoost multi-stage predictions saved to {output_file}")
print(f"Total predictions: {len(df_pred)}")
print(f"Unstable cases (with regression): {mask_failure.sum()}")
print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
