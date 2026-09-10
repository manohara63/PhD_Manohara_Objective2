import warnings
import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Suppress warnings and LightGBM logs for clean output
warnings.filterwarnings('ignore')
os.environ['LIGHTGBM_VERBOSE'] = '0'

# --- Load and Clean Data ---
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Classification input_refined.xlsx'
data = pd.read_excel(file_path)
data = data.dropna(how='all')  # Remove empty rows

# --- Features and Target ---
X = data.drop(columns=['Slope state', 'Slope type', 'No of days for failure', 'PWP at failure'], errors='ignore')
y = data['Slope state']

categorical_features = ['Rainfall realisation']

# --- Ensure Categorical Columns Are Categorical Dtype ---
for col in categorical_features:
    X[col] = X[col].astype('category')

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# --- Build Model and Hyperparameter Grid ---
lgbm_model = lgb.LGBMClassifier(
    objective='binary' if y.nunique() == 2 else 'multiclass',
    random_state=42,
    n_jobs=-1,
    verbose=-1  # Suppress LightGBM output
)

param_dist = {
    'num_leaves': [15, 31, 63],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [200, 500, 1000],
    'min_child_samples': [10, 20, 50],
    'reg_alpha': [0, 1, 3, 5],
    'reg_lambda': [0, 1, 3, 5],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=lgbm_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=2,        # This is for sklearn, keep for fit reporting
    random_state=42,
    n_jobs=-1,
    return_train_score=True
)

# --- Fit Model ---
random_search.fit(X_train, y_train, categorical_feature=categorical_features)

# --- Print Results ---
print("Best hyperparameters:", random_search.best_params_)
print("Best CV Accuracy: {:.4f}".format(random_search.best_score_))

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nTest Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nTraining accuracy: {train_accuracy:.4f}")

# --- Optional: Write Results to File for Large Output ---
with open('results_LGBM.txt', 'w') as f:
    f.write("Best hyperparameters: {}\n".format(random_search.best_params_))
    f.write("Best CV Accuracy: {:.4f}\n".format(random_search.best_score_))
    f.write("\nTest Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))
    f.write("\nTraining accuracy: {:.4f}\n".format(train_accuracy))
