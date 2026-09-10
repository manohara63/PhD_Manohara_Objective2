import warnings
import os
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# --- Load and Clean Data ---
file_path = r'D:\PhD work\Objective 2\FE modelling\ML Modelling\Trial 2_additional inputs\Obj2_T2_Classification input_refined.xlsx'
data = pd.read_excel(file_path)
data = data.dropna(how='all')

# --- Features and Target (NO DATA LEAKAGE) ---
X = data.drop(columns=['Slope state', 'Slope type', 'No of days for failure', 'PWP at failure'], errors='ignore')
y = data['Slope state']

print(f"Original input features: {X.shape[1]} columns")
print(f"Classes: {y.unique()}")

# --- FIXED: Precise column matching ---
categorical_features = ['Rainfall realisation']
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)
print(f"After one-hot encoding: {X_encoded.shape[1]} features")

# FIXED: Only match EXACT rainfall realisation dummies
rainfall_dummies = [col for col in X_encoded.columns if col.startswith('Rainfall realisation_')]
print(f"Pure Rainfall dummies ({len(rainfall_dummies)}): {rainfall_dummies[:5]}...")

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, stratify=y, random_state=42)

# --- FIXED: Use 'estimator' parameter (not base_estimator) ---
param_dist = {
    # Core AdaBoost parameters
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    'algorithm': ['SAMME', 'SAMME.R'],
    
    # FIXED: Base Estimator parameters - CORRECT 'estimator__' PREFIX
    'estimator__max_depth': [1, 2, 3],
    'estimator__min_samples_split': [2, 5, 10, 20],
    'estimator__min_samples_leaf': [1, 2, 4, 8],
    'estimator__max_features': ['sqrt', 'log2', None],
    'estimator__min_impurity_decrease': [0.0, 0.001, 0.01]
}

# --- FIXED: Specify estimator explicitly ---
estimator_template = DecisionTreeClassifier(random_state=42)

adaboost_model = AdaBoostClassifier(
    estimator=estimator_template,  # ✅ CORRECT parameter name
    random_state=42
)

# --- RandomizedSearchCV ---
print("\n" + "="*70)
print("STARTING FIXED ADABOOST TUNING (R1 INCLUDED)")
print("="*70)

random_search = RandomizedSearchCV(
    estimator=adaboost_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    return_train_score=True
)

# --- Fit Model ---
random_search.fit(X_train, y_train)

# --- Print Results ---
print("\n" + "="*80)
print("✅ FIXED ADABOOST RESULTS")
print("="*80)
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
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTraining accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Train-Test Gap: {train_accuracy - test_accuracy:.4f}")

# --- Save Results ---
with open('results_ADABOOST_FIXED.txt', 'w') as f:
    f.write("FIXED ADABOOST TUNING RESULTS\n")
    f.write(f"Features: {X_encoded.shape[1]} | Rainfall dummies: {len(rainfall_dummies)}\n\n")
    f.write(f"Best hyperparameters: {random_search.best_params_}\n")
    f.write(f"Best CV Accuracy: {random_search.best_score_:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))
    f.write(f"\nTrain Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")

print(f"\n✅ Results saved to 'results_ADABOOST_FIXED.txt'")
