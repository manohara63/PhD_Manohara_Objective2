import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Load data
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Classification input_refined.xlsx'
data = pd.read_excel(file_path)

# Features and target
X = data.drop(columns=['Slope state'])
y = data['Slope state']

categorical_features = ['Rainfall realisation']

# Initial train-test split (70-30 stratified)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Further split training into train (80%) and valid (20%) stratified for early stopping
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

# Prepare CatBoost Pools (required for early stopping and categorical features)
train_pool = Pool(X_train, label=y_train, cat_features=categorical_features)
valid_pool = Pool(X_valid, label=y_valid, cat_features=categorical_features)

# Define CatBoostClassifier with silent output for RandomizedSearchCV
cat_model = CatBoostClassifier(
    silent=True,
    random_seed=42,
    cat_features=categorical_features
)

# Hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'iterations': [200, 500, 1000],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 50, 100],
    'random_strength': [0, 1, 2, 3]
}

# Custom fit function to support early stopping by wrapping CatBoost
def fit_with_early_stopping(self, X, y=None, **kwargs):
    # Use train and validation pools created externally
    return CatBoostClassifier.fit(self, train_pool, eval_set=valid_pool, early_stopping_rounds=50, **kwargs)

# Patch fit method
cat_model.fit = fit_with_early_stopping.__get__(cat_model)

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV (early stopping used internally via patched fit)
random_search.fit(X_train, y_train)

print("Best hyperparameters:", random_search.best_params_)
print("Best CV Accuracy: {:.4f}".format(random_search.best_score_))

# Evaluate on the test set with best estimator
best_model = random_search.best_estimator_

test_pool = Pool(X_test, label=y_test, cat_features=categorical_features)
y_pred = best_model.predict(X_test)

print("Test Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# After you have your best_model trained

y_train_pred = best_model.predict(X_train)
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Training accuracy: {train_accuracy:.4f}')
