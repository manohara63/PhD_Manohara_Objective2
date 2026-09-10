import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product

# Load data
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Regression input_refined.xlsx'
data = pd.read_excel(file_path).dropna(how='any')

# Prepare inputs and multi-target outputs
X = data.drop(columns=['No of days for failure', 'PWP at failure'])
y = data[['No of days for failure', 'PWP at failure']]

# One-hot encode categorical features for AdaBoost
X_encoded = pd.get_dummies(X, columns=['Rainfall realisation'], drop_first=False)

# Train-test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42)

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42)

# Hyperparameter grid for manual search (AdaBoost-specific)
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    'estimator_max_depth': [1, 2, 3, 4],
    'estimator_min_samples_split': [2, 5, 10],
    'estimator_min_samples_leaf': [1, 2, 4]
}

best_score = 0  # maximize R2
best_params_per_target = {}
best_models = {}

print("Starting manual hyperparameter search for AdaBoost without early stopping...")

# Loop over each target separately
for target_idx, target_name in enumerate(y.columns):
    print(f"\nTuning for target: {target_name}")
    y_train_target = y_train.iloc[:, target_idx]
    y_valid_target = y_valid.iloc[:, target_idx]

    best_score_target = -np.inf  # maximize R2
    best_params_target = None
    best_model_target = None

    for params in product(*param_grid.values()):
        params_dict = dict(zip(param_grid.keys(), params))
        
        # ✅ FIXED: Create estimator with renamed parameters
        estimator = DecisionTreeRegressor(
            max_depth=params_dict['estimator_max_depth'],
            min_samples_split=params_dict['estimator_min_samples_split'],
            min_samples_leaf=params_dict['estimator_min_samples_leaf'],
            random_state=42
        )
        
        # Core AdaBoost parameters only
        adaboost_params = {
            'n_estimators': params_dict['n_estimators'],
            'learning_rate': params_dict['learning_rate'],
            'estimator': estimator  # ✅ FIXED: Use 'estimator' not 'base_estimator'
        }
        
        model = AdaBoostRegressor(random_state=42, **adaboost_params)
        model.fit(X_train, y_train_target)
        
        preds_valid = model.predict(X_valid)
        r2 = r2_score(y_valid_target, preds_valid)
        print(f"Params: {params_dict}, Validation R2: {r2:.4f}")
        
        if r2 > best_score_target:
            best_score_target = r2
            best_params_target = params_dict
            best_model_target = model

    best_score += best_score_target
    best_params_per_target[target_name] = best_params_target
    best_models[target_name] = best_model_target

print("\nBest hyperparameters per target:")
for t in y.columns:
    print(f"{t}: {best_params_per_target[t]}")

print(f"\nSum of best validation R2 scores: {best_score:.4f}")

# Evaluate on test set
print("\nTest set performance:")
for target_idx, target_name in enumerate(y.columns):
    model = best_models[target_name]
    y_test_target = y_test.iloc[:, target_idx]
    preds_test = model.predict(X_test)
    rmse = mean_squared_error(y_test_target, preds_test, squared=False)
    mae = mean_absolute_error(y_test_target, preds_test)
    r2 = r2_score(y_test_target, preds_test)
    print(f"{target_name} Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
