import os
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product

# Load data
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Regression input_refined.xlsx'
data = pd.read_excel(file_path).dropna(how='any')

# Prepare inputs and multi-target outputs
X = data.drop(columns=['No of days for failure', 'PWP at failure'])
y = data[['No of days for failure', 'PWP at failure']]

categorical_features = ['Rainfall realisation']
for c in categorical_features:
    X[c] = X[c].astype('category')

# Train-test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42)

# Hyperparameter grid for manual search
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.03, 0.1],
    'iterations': [50, 100, 200, 500, 1000, 1500, 2000],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 50],
    'random_strength': [0, 1, 2]
}

best_params_per_target = {}
best_models = {}

print("Starting manual hyperparameter search per target...")

for target_name in y.columns:
    print(f"\nTuning for target: {target_name}")

    y_train_target = y_train[target_name]
    y_valid_target = y_valid[target_name]

    train_pool = Pool(X_train, y_train_target, cat_features=categorical_features)
    valid_pool = Pool(X_valid, y_valid_target, cat_features=categorical_features)

    best_score = float('inf')
    best_params = None
    best_model = None

    for params in product(*param_grid.values()):
        params_dict = dict(zip(param_grid.keys(), params))
        model = CatBoostRegressor(
            random_seed=42,
            silent=True,
            cat_features=categorical_features,
            loss_function='RMSE',
            **params_dict
        )
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)
        preds_valid = model.predict(X_valid)
        rmse = mean_squared_error(y_valid_target, preds_valid, squared=False)
        print(f"Params: {params_dict}, Validation RMSE: {rmse:.4f}")
        if rmse < best_score:
            best_score = rmse
            best_params = params_dict
            best_model = model

    best_params_per_target[target_name] = best_params
    best_models[target_name] = best_model

print("\nBest hyperparameters per target:")
for t in y.columns:
    print(f"{t}: {best_params_per_target[t]}")

# Evaluate on test set
print("\nTest set performance:")
for target_name in y.columns:
    model = best_models[target_name]
    preds_test = model.predict(X_test)
    rmse = mean_squared_error(y_test[target_name], preds_test, squared=False)
    mae = mean_absolute_error(y_test[target_name], preds_test)
    r2 = r2_score(y_test[target_name], preds_test)
    print(f"{target_name} Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
