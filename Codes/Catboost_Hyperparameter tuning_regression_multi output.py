import os
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product

# Load data
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Obj2_Regression input.xlsx'
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

train_pool = Pool(X_train, y_train, cat_features=categorical_features)
valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)

# Hyperparameter grid for manual search
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.03, 0.1],
    'iterations': [50, 100, 200, 500, 1000, 1500, 2000],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 50],
    'random_strength': [0, 1, 2]
}

best_score = float('inf')
best_params = None
best_model = None

print("Starting manual hyperparameter search...")

for params in product(*param_grid.values()):
    params_dict = dict(zip(param_grid.keys(), params))
    model = CatBoostRegressor(
        random_seed=42,
        silent=True,
        cat_features=categorical_features,
        loss_function='MultiRMSE',
        **params_dict
    )
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)
    preds_valid = model.predict(X_valid)
    rmse = mean_squared_error(y_valid, preds_valid, squared=False)
    print(f"Params: {params_dict}, Validation RMSE: {rmse:.4f}")
    if rmse < best_score:
        best_score = rmse
        best_params = params_dict
        best_model = model

print("Best hyperparameters found:", best_params)
print(f"Validation RMSE for optimized model: {best_score:.4f}")

# Evaluate on test set
preds_test = best_model.predict(X_test)
for i, target_name in enumerate(y.columns):
    rmse = mean_squared_error(y_test.iloc[:, i], preds_test[:, i], squared=False)
    mae = mean_absolute_error(y_test.iloc[:, i], preds_test[:, i])
    r2 = r2_score(y_test.iloc[:, i], preds_test[:, i])
    print(f"{target_name} Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
