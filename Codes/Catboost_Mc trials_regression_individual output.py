import pandas as pd
import numpy as np
import warnings
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Regression input_refined.xlsx'
data = pd.read_excel(file_path).dropna(how='any')

X = data.drop(columns=['No of days for failure', 'PWP at failure'])
y = data[['No of days for failure', 'PWP at failure']]

categorical_features = ['Rainfall realisation']
for c in categorical_features:
    X[c] = X[c].astype('category')

# ✅ BEST HYPERPARAMETERS from GridSearchCV (updated)
params_no_days = {
    'depth': 4,                    
    'learning_rate': 0.1,          
    'iterations': 500,             
    'l2_leaf_reg': 1,              
    'border_count': 32,            
    'random_strength': 1,          
    'loss_function': 'RMSE',
    'silent': True,
    'cat_features': categorical_features,
    'random_seed': 42             
}

params_pwp = {
    'depth': 4,                    
    'learning_rate': 0.1,          
    'iterations': 200,             
    'l2_leaf_reg': 1,             
    'border_count': 32,            
    'random_strength': 1,         
    'loss_function': 'RMSE',
    'silent': True,
    'cat_features': categorical_features,
    'random_seed': 42              
}

# Containers for metrics
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

print("Dataset shape:", X.shape)
print("\n✅ Using BEST HYPERPARAMETERS from GridSearchCV:")
print("No of days for failure:", params_no_days)
print("PWP at failure:", params_pwp)

for i in range(100):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i)
    
    # --- No of days for failure (BEST HP) ---
    model_days = CatBoostRegressor(**params_no_days)
    model_days.fit(X_train, y_train['No of days for failure'])
    pred_train_days = model_days.predict(X_train)
    pred_test_days = model_days.predict(X_test)
    
    results['train_rmse_no_days'].append(mean_squared_error(
        y_train['No of days for failure'], pred_train_days, squared=False))
    results['test_rmse_no_days'].append(mean_squared_error(
        y_test['No of days for failure'], pred_test_days, squared=False))
    results['train_mae_no_days'].append(mean_absolute_error(
        y_train['No of days for failure'], pred_train_days))
    results['test_mae_no_days'].append(mean_absolute_error(
        y_test['No of days for failure'], pred_test_days))
    results['train_r2_no_days'].append(r2_score(
        y_train['No of days for failure'], pred_train_days))
    results['test_r2_no_days'].append(r2_score(
        y_test['No of days for failure'], pred_test_days))
    
    # --- PWP at failure (BEST HP) ---
    model_pwp = CatBoostRegressor(**params_pwp)
    model_pwp.fit(X_train, y_train['PWP at failure'])
    pred_train_pwp = model_pwp.predict(X_train)
    pred_test_pwp = model_pwp.predict(X_test)
    
    results['train_rmse_pwp'].append(mean_squared_error(
        y_train['PWP at failure'], pred_train_pwp, squared=False))
    results['test_rmse_pwp'].append(mean_squared_error(
        y_test['PWP at failure'], pred_test_pwp, squared=False))
    results['train_mae_pwp'].append(mean_absolute_error(
        y_train['PWP at failure'], pred_train_pwp))
    results['test_mae_pwp'].append(mean_absolute_error(
        y_test['PWP at failure'], pred_test_pwp))
    results['train_r2_pwp'].append(r2_score(
        y_train['PWP at failure'], pred_train_pwp))
    results['test_r2_pwp'].append(r2_score(
        y_test['PWP at failure'], pred_test_pwp))

df_results = pd.DataFrame(results)
print("\nSummary metrics over 100 repeated runs (BEST HYPERPARAMETERS):")
print(df_results.agg(['mean', 'std']))

# Create output directory and save
output_dir = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\CatBoost'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'CatBoost_regression_100runs_refined.xlsx')
df_results.to_excel(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")
