import pandas as pd
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Classification input_refined.xlsx'
data = pd.read_excel(file_path)
data = data.dropna(how='all')

# Features and target (NO DATA LEAKAGE)
X = data.drop(columns=['Slope state', 'Slope type', 'No of days for failure', 'PWP at failure'], errors='ignore')
y = data['Slope state']

categorical_features = ['Rainfall realisation']
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)  # ALL R1-R20 included

# ✅ UPDATED: Best hyperparameters from GridSearchCV
estimator = DecisionTreeClassifier(
    max_depth=3,                    # estimator__max_depth: 3
    min_samples_split=2,            # estimator__min_samples_split: 2
    min_samples_leaf=8,             # estimator__min_samples_leaf: 8
    min_impurity_decrease=0.001,    # estimator__min_impurity_decrease: 0.001
    max_features='log2',            # estimator__max_features: 'log2'
    random_state=42
)

params_adaboost = {
    'n_estimators': 200,        # n_estimators: 200
    'learning_rate': 0.2,       # learning_rate: 0.2
    'algorithm': 'SAMME.R',     # algorithm: 'SAMME.R'
    'estimator': estimator,
    'random_state': 42
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

# Map classes to numbers for roc_auc calculation
class_map = {'Stable':0, 'Unstable':1}

print(f"Dataset: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features")
print("Using BEST HYPERPARAMETERS from GridSearchCV:")
print(f"n_estimators: {params_adaboost['n_estimators']}")
print(f"learning_rate: {params_adaboost['learning_rate']}")
print(f"estimator params: max_depth={estimator.max_depth}, min_samples_split={estimator.min_samples_split}, min_samples_leaf={estimator.min_samples_leaf}")

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, stratify=y, random_state=i)

    # Create NEW model instance each iteration
    model = AdaBoostClassifier(**params_adaboost)
    model.fit(X_train, y_train)

    # Predict and predict probability for AUC
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    results['train_accuracy'].append(accuracy_score(y_train, y_train_pred))
    results['test_accuracy'].append(accuracy_score(y_test, y_test_pred))

    # Calculate precision, recall, f1 for both classes
    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, labels=['Stable', 'Unstable'])
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, labels=['Stable', 'Unstable'])
    
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

    # Calculate AUC
    results['train_auc'].append(roc_auc_score(y_train.map(class_map), y_train_proba))
    results['test_auc'].append(roc_auc_score(y_test.map(class_map), y_test_proba))

df_results = pd.DataFrame(results)
print("\nSummary metrics over 100 runs (BEST HYPERPARAMETERS ADABOOST):")
print(df_results.agg(['mean', 'std']))

# Create output directory if needed
output_dir = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\AdaBoost'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'AdaBoost_classification_metrics_100runs_refined.xlsx')
df_results.to_excel(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")
