import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV

# 1. Load the extracted features from CSV files
train_features = pd.read_csv('ecg_cardiac_features_train.csv')
val_features = pd.read_csv('ecg_cardiac_features_val.csv')
test_features = pd.read_csv('ecg_cardiac_features_test.csv')

# Choose one class as the positive class (e.g., the non-NORM class)
y_train = pd.read_csv('Y_train_1.csv', index_col=0)
y_val = pd.read_csv('Y_valid_1.csv', index_col=0)
y_test = pd.read_csv('Y_test_1.csv', index_col=0)

# For binary classification, choose the second column (e.g., 'MI', 'STTC', or 'CD')
# Assumes second column is the positive class of interest
y_train = y_train.iloc[:, 1]
y_val = y_val.iloc[:, 1]
y_test = y_test.iloc[:, 1]

# 3. Extract patient IDs and prepare feature matrices
# Store patient IDs for reference
train_patient_ids = train_features['patient_id']
val_patient_ids = val_features['patient_id']
test_patient_ids = test_features['patient_id']

# 4. Drop non-feature columns
X_train = train_features.drop('patient_id', axis=1)
X_val = val_features.drop('patient_id', axis=1)
X_test = test_features.drop('patient_id', axis=1)

# 5. Impute missing values (NaN)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

print(f"Training data shape: {X_train_imputed.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val_imputed.shape}")
print(f"Validation labels shape: {y_val.shape}")
print(f"Test data shape: {X_test_imputed.shape}")
print(f"Test labels shape: {y_test.shape}")

# 6. Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 7. Create and train SVM model with default parameters
svm_model = SVC(
    probability=True,
    class_weight='balanced',
    random_state=42
)

# 8. Train the model
print("Training initial SVM model...")
svm_model.fit(X_train_scaled, y_train)

# 9. Evaluate on validation set
val_predictions = svm_model.predict(X_val_scaled)
val_proba = svm_model.predict_proba(X_val_scaled)[:, 1]

print("\nInitial Validation Performance:")
print(confusion_matrix(y_val, val_predictions))
print(classification_report(y_val, val_predictions))
print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

# 10. OPTIMIZATION: Faster hyperparameter tuning with these methods:
# - Using RandomizedSearchCV instead of GridSearchCV
# - Using a subset of the training data
# - Reducing the number of cross-validation folds
# - Focusing on a more targeted parameter space

print("\nPerforming randomized search for faster hyperparameter tuning...")

# OPTION 1: If your dataset is large, use a subset for tuning
# Determine if downsampling would be beneficial (e.g., if dataset > 5000 samples)
subset_size = min(5000, X_train_scaled.shape[0])
if X_train_scaled.shape[0] > subset_size:
    print(f"Using a subset of {subset_size} samples for hyperparameter tuning...")
    indices = np.random.choice(X_train_scaled.shape[0], subset_size, replace=False)
    X_train_subset = X_train_scaled[indices]
    y_train_subset = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
else:
    X_train_subset = X_train_scaled
    y_train_subset = y_train

# Define a focused parameter space
param_distributions = {
    'C': [0.1, 1, 10, 100],  # Cost parameter
    'gamma': ['scale', 'auto'],  # Reduced gamma options
    'kernel': ['rbf', 'linear']  # Focused on most common kernels
}

# Use RandomizedSearchCV with fewer iterations and CV folds
random_search = RandomizedSearchCV(
    SVC(probability=True, class_weight='balanced', random_state=42),
    param_distributions=param_distributions,
    n_iter=8,  # Try 8 parameter combinations instead of all combinations
    cv=3,      # Use 3-fold CV instead of 5-fold to speed up
    scoring='roc_auc',
    n_jobs=-1,  # Use all available processors
    random_state=42
)

# Fit the random search
random_search.fit(X_train_subset, y_train_subset)
print(f"Best parameters: {random_search.best_params_}")
best_svm_model = random_search.best_estimator_

# 11. Evaluate tuned model on validation set
tuned_val_predictions = best_svm_model.predict(X_val_scaled)
tuned_val_proba = best_svm_model.predict_proba(X_val_scaled)[:, 1]

print("\nTuned Model Validation Performance:")
print(confusion_matrix(y_val, tuned_val_predictions))
print(classification_report(y_val, tuned_val_predictions))
print(f"ROC-AUC: {roc_auc_score(y_val, tuned_val_proba):.4f}")

# 12. Final evaluation on test set
test_predictions = best_svm_model.predict(X_test_scaled)
test_proba = best_svm_model.predict_proba(X_test_scaled)[:, 1]

print("\nTest Performance:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print(f"ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")

# 13. Feature importance for SVM model (only applicable for linear kernel)
if best_svm_model.kernel == 'linear':
    feature_names = X_train.columns
    coefficients = best_svm_model.coef_[0]
    importance = np.abs(coefficients)

    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    print("\nTop 20 most important features (for linear kernel):")
    print(feature_importance.head(20))
else:
    print("\nFeature importance is only directly available for linear kernel SVMs.")
    print(f"Current model uses '{best_svm_model.kernel}' kernel.")

# 14. Save the trained model and preprocessing objects
import joblib

print("\nSaving model and preprocessing objects...")
joblib.dump(best_svm_model, 'ecg_svm_model.pkl')
print("Model and preprocessing objects saved successfully.")
