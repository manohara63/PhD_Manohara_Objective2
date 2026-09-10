import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Obj2_Regression input.xlsx'
data = pd.read_excel(file_path).dropna(how='any')

X = data.drop(columns=['No of days for failure', 'PWP at failure'])
y = data[['No of days for failure', 'PWP at failure']]

categorical_features = ['Rainfall realisation']
for c in categorical_features:
    X[c] = X[c].astype('category')

catboost_params = {
    'depth': 8,
    'learning_rate': 0.1,
    'iterations': 50,
    'l2_leaf_reg': 1,
    'border_count': 32,
    'random_strength': 0,
    'loss_function': 'MultiRMSE',
    'silent': True,
    'cat_features': categorical_features
}

# Results containers
results = {
    'train_rmse_no_days': [],
    'test_rmse_no_days': [],
    'train_mae_no_days': [],
    'test_mae_no_days': [],
    'train_r2_no_days': [],
    'test_r2_no_days': [],
    'train_rmse_pwp': [],
    'test_rmse_pwp': [],
    'train_mae_pwp': [],
    'test_mae_pwp': [],
    'train_r2_pwp': [],
    'test_r2_pwp': []
}

for i in range(100):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i)
    
    # Multi-output regression
    model = CatBoostRegressor(**catboost_params)
    model.fit(X_train, y_train)
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    # No of days for failure (column index 0)
    results['train_rmse_no_days'].append(mean_squared_error(y_train.iloc[:, 0], pred_train[:, 0], squared=False))
    results['test_rmse_no_days'].append(mean_squared_error(y_test.iloc[:, 0], pred_test[:, 0], squared=False))
    results['train_mae_no_days'].append(mean_absolute_error(y_train.iloc[:, 0], pred_train[:, 0]))
    results['test_mae_no_days'].append(mean_absolute_error(y_test.iloc[:, 0], pred_test[:, 0]))
    results['train_r2_no_days'].append(r2_score(y_train.iloc[:, 0], pred_train[:, 0]))
    results['test_r2_no_days'].append(r2_score(y_test.iloc[:, 0], pred_test[:, 0]))
    
    # PWP at failure (column index 1)
    results['train_rmse_pwp'].append(mean_squared_error(y_train.iloc[:, 1], pred_train[:, 1], squared=False))
    results['test_rmse_pwp'].append(mean_squared_error(y_test.iloc[:, 1], pred_test[:, 1], squared=False))
    results['train_mae_pwp'].append(mean_absolute_error(y_train.iloc[:, 1], pred_train[:, 1]))
    results['test_mae_pwp'].append(mean_absolute_error(y_test.iloc[:, 1], pred_test[:, 1]))
    results['train_r2_pwp'].append(r2_score(y_train.iloc[:, 1], pred_train[:, 1]))
    results['test_r2_pwp'].append(r2_score(y_test.iloc[:, 1], pred_test[:, 1]))

df_results = pd.DataFrame(results)
print("Summary metrics over 100 repeated runs:")
print(df_results.agg(['mean', 'std']))

df_results.to_excel(r'D:\PhD work\Objective 2\FE modelling\ML Modelling\CatBoost\CatBoost_REG_multioutput_100runs.xlsx', index=False)
