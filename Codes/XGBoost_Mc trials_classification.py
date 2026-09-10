import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Classification input.xlsx'
data = pd.read_excel(file_path)
data = data.dropna(how='all')

# Features and target (NO DATA LEAKAGE)
X = data.drop(columns=['Slope state', 'Slope type', 'No of days for failure', 'PWP at failure'], errors='ignore')
y = data['Slope state']

# Encode categorical features and labels
categorical_features = ['Rainfall realisation']
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)

# Label encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TUNED HYPERPARAMETERS from RandomizedSearchCV
params_xgboost = {
    'subsample': 1.0,
    'reg_lambda': 0.1,
    'reg_alpha': 0,
    'n_estimators': 200,
    'min_child_weight': 1,
    'max_depth': 5,
    'learning_rate': 0.2,
    'gamma': 0.3,
    'colsample_bytree': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'random_state': 42,
    'verbosity': 0
}

# Containers for metrics (EXACT SAME as CatBoost)
results = {
    'train_accuracy': [],
    'test_accuracy': [],
    'train_precision_stable': [],
    'train_recall_stable': [],
    'train_f1_stable': [],
    'test_precision_stable': [],
    'test_recall_stable': [],
    'test_f1_stable': [],
    'train_precision_unstable': [],
    'train_recall_unstable': [],
    'train_f1_unstable': [],
    'test_precision_unstable': [],
    'test_recall_unstable': [],
    'test_f1_unstable': [],
    'train_auc': [],
    'test_auc': []
}

class_map = {0: 'Stable', 1: 'Unstable'}  # For reporting

print(f"Dataset: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features")
print(f"Tuned XGBoost - Running 100 iterations...")

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.3, stratify=y_encoded, random_state=i)

    # Create and fit tuned XGBoost model
    model = XGBClassifier(**params_xgboost)
    model.fit(X_train, y_train)

    # Predict and predict probability for AUC
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    results['train_accuracy'].append(accuracy_score(y_train, y_train_pred))
    results['test_accuracy'].append(accuracy_score(y_test, y_test_pred))

    # Decode for precision/recall/f1 calculation
    y_train_labels = label_encoder.inverse_transform(y_train)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_train_pred_labels = label_encoder.inverse_transform(y_train_pred)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

    # Calculate precision, recall, f1 for both classes
    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_train_labels, y_train_pred_labels, labels=['Stable', 'Unstable'])
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test_labels, y_test_pred_labels, labels=['Stable', 'Unstable'])
    
    # stable index is 0, unstable index is 1
    results['train_precision_stable'].append(train_prec[0])
    results['train_recall_stable'].append(train_rec[0])
    results['train_f1_stable'].append(train_f1[0])
    results['train_precision_unstable'].append(train_prec[1])
    results['train_recall_unstable'].append(train_rec[1])
    results['train_f1_unstable'].append(train_f1[1])
    
    results['test_precision_stable'].append(test_prec[0])
    results['test_recall_stable'].append(test_rec[0])
    results['test_f1_stable'].append(test_f1[0])
    results['test_precision_unstable'].append(test_prec[1])
    results['test_recall_unstable'].append(test_rec[1])
    results['test_f1_unstable'].append(test_f1[1])

    # Calculate AUC (numeric labels)
    results['train_auc'].append(roc_auc_score(y_train, y_train_proba))
    results['test_auc'].append(roc_auc_score(y_test, y_test_proba))

    if i % 20 == 0:
        print(f"Completed {i+1}/100 runs...")

df_results = pd.DataFrame(results)
print("\n" + "="*60)
print("XGBoost TUNED - Summary metrics over 100 runs:")
print("="*60)
print(df_results.agg(['mean', 'std']))

# Create output directory
out_dir = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\XGBoost'
os.makedirs(out_dir, exist_ok=True)
excel_path = os.path.join(out_dir, 'XGBoost_classification_metrics_100runs_refined.xlsx')
df_results.to_excel(excel_path, index=False)

print(f"\n✅ Results saved to: {excel_path}")
print("📊 Ready for comparison with CatBoost/LightGBM/AdaBoost!")
