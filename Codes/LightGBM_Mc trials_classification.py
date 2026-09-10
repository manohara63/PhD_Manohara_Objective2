import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Load data
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Classification input_refined.xlsx'
data = pd.read_excel(file_path)
data = data.dropna(how='all')

# Features and target
X = data.drop(columns=['Slope state', 'Slope type', 'No of days for failure', 'PWP at failure'], errors='ignore')
y = data['Slope state']

categorical_features = ['Rainfall realisation']
for col in categorical_features:
    X[col] = X[col].astype('category')

# TUNED HYPERPARAMETERS from RandomizedSearchCV
params_lgbm = {
    'reg_lambda': 5,          # Updated
    'reg_alpha': 0,           # Updated: was 3
    'num_leaves': 63,         # Updated: was 31
    'n_estimators': 1000,      # Same
    'min_child_samples': 10,  # Updated: was 20
    'max_depth': 8,           # Same
    'learning_rate': 0.05,    # Updated: was 0.1
    'colsample_bytree': 0.8,  # Updated: was 0.6
    'objective': 'binary',
    'verbose': -1
}

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
    'test_auc': [],
}

class_map = {'Stable': 0, 'Unstable': 1}

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=i)

    model = lgb.LGBMClassifier(**params_lgbm)
    model.fit(X_train, y_train, categorical_feature=categorical_features)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    results['train_accuracy'].append(accuracy_score(y_train, y_train_pred))
    results['test_accuracy'].append(accuracy_score(y_test, y_test_pred))

    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, labels=['Stable', 'Unstable'])
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, labels=['Stable', 'Unstable'])

    results['train_precision_stable'].append(train_prec[0])
    results['train_recall_stable'].append(train_rec[0])
    results['train_f1_stable'].append(train_f1[0])
    results['test_precision_stable'].append(test_prec[0])
    results['test_recall_stable'].append(test_rec[0])
    results['test_f1_stable'].append(test_f1[0])

    results['train_precision_unstable'].append(train_prec[1])
    results['train_recall_unstable'].append(train_rec[1])
    results['train_f1_unstable'].append(train_f1[1])
    results['test_precision_unstable'].append(test_prec[1])
    results['test_recall_unstable'].append(test_rec[1])
    results['test_f1_unstable'].append(test_f1[1])

    results['train_auc'].append(roc_auc_score(y_train.map(class_map), y_train_proba))
    results['test_auc'].append(roc_auc_score(y_test.map(class_map), y_test_proba))

df_results = pd.DataFrame(results)
print("LightGBM TUNED summary metrics over 100 runs:")
print(df_results.agg(['mean', 'std']))

df_results.to_excel(r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\LightGBM\LightGBM_classification_metrics_100runs_refined.xlsx', index=False)
print("Tuned LightGBM results saved!")
