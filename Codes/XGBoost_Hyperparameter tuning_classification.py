import warnings
import os
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Load and Clean Data ---
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Classification input.xlsx'
data = pd.read_excel(file_path)
data = data.dropna(how='all')

# --- Features and Target (NO DATA LEAKAGE) ---
X = data.drop(columns=['Slope state', 'Slope type', 'No of days for failure', 'PWP at failure'], errors='ignore')
y = data['Slope state']

# ✅ FIXED: Convert string labels to numeric (0,1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

print(f"Original input features: {X.shape[1]} columns")
print(f"Classes: {np.unique(y_encoded)} (encoded)")

# --- One-hot encode categorical feature(s) ---
categorical_features = ['Rainfall realisation']
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)
print(f"After one-hot encoding: {X_encoded.shape[1]} features")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)

# --- XGBoost Model ---
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    n_jobs=-1,
    random_state=42,
    verbosity=0  # Suppress XGBoost warnings
)

# --- Hyperparameter Search Space ---
param_dist = {
    'max_depth': [3, 4, 5, 6, 8],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1],
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

# --- RandomizedSearchCV ---
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    return_train_score=True,
    error_score='raise'  # Show exact errors if any occur
)

print("\n===================================================")
print("Starting XGBoost RandomizedSearchCV (LABELS ENCODED)")
print("===================================================")

random_search.fit(X_train, y_train)

# --- Results ---
print("\n================ XGBOOST RESULTS ================")
print("Best hyperparameters:", random_search.best_params_)
print("Best CV Accuracy: {:.4f}".format(random_search.best_score_))

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Decode predictions back to original labels for reporting
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\nTest Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))
print("Confusion Matrix:")
print(confusion_matrix(y_test_labels, y_pred_labels))

# Train accuracy
y_train_pred = best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTraining accuracy: {train_accuracy:.4f}")
print(f"Test accuracy:     {test_accuracy:.4f}")
print(f"Train-Test Gap:    {train_accuracy - test_accuracy:.4f}")

# --- Save Results ---
out_dir = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\XGBoost'
os.makedirs(out_dir, exist_ok=True)
txt_path = os.path.join(out_dir, 'XGBoost_RandomSearch_results.txt')

with open(txt_path, 'w') as f:
    f.write("XGBoost RandomizedSearchCV Results\n")
    f.write("="*50 + "\n\n")
    f.write(f"Best hyperparameters: {random_search.best_params_}\n")
    f.write(f"Best CV Accuracy: {random_search.best_score_:.4f}\n\n")
    f.write("Test Classification Report:\n")
    f.write(classification_report(y_test_labels, y_pred_labels))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test_labels, y_pred_labels)))
    f.write(f"\n\nTraining accuracy: {train_accuracy:.4f}\n")
    f.write(f"Test accuracy: {test_accuracy:.4f}\n")

print(f"\n✅ Results saved to: {txt_path}")
