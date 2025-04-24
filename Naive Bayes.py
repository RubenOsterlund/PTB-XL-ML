import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

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
y_train = y_train.iloc[:, 1]  # Take the second column directly
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

feature_names = X_train.columns

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

# 7. Create and perform GridSearch for GaussianNB
print("Performing GridSearch on Gaussian Naive Bayes model...")

# Define parameter grid for GaussianNB
param_grid = {
    'var_smoothing': np.logspace(-11, -5, 10),  # Range of smoothing variance values
    'priors': [None, [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]  # Different class prior probabilities
}

# Create base model
nb_base = GaussianNB()

# Create and run GridSearchCV
# Using stratified 5-fold cross-validation since this appears to be an imbalanced classification problem
grid_search = GridSearchCV(
    estimator=nb_base,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',  # Using ROC-AUC as the scoring metric since you're using it for evaluation
    verbose=1,
    n_jobs=-1  # Use all available cores
)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Print GridSearch results
print("\nGrid Search Results:")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Get the best model
best_nb_model = grid_search.best_estimator_

# 8. Evaluate on validation set using best model
val_predictions = best_nb_model.predict(X_val_scaled)
val_proba = best_nb_model.predict_proba(X_val_scaled)[:, 1]  # For ROC-AUC

print("\nValidation Performance:")
print(confusion_matrix(y_val, val_predictions))
print(classification_report(y_val, val_predictions))
print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")


# 9. Evaluate on test set
test_predictions = best_nb_model.predict(X_test_scaled)
test_proba = best_nb_model.predict_proba(X_test_scaled)[:, 1]

print("\nTest Performance:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print(f"ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")


# 10. Analyze feature importance for best Naive Bayes model
feature_names = np.array(feature_names)
class_0_means = best_nb_model.theta_[0]  # Mean for class 0
class_1_means = best_nb_model.theta_[1]  # Mean for class 1
class_variances = best_nb_model.var_  # Variance for each feature

# Calculate absolute difference in means, scaled by variance
mean_diffs = np.abs(class_1_means - class_0_means)
importance = mean_diffs / np.sqrt(np.mean(class_variances, axis=0))

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nTop 20 most important features for best Naive Bayes model:")
print(feature_importance.head(20))

# 11. Save the best model and preprocessing objects
import joblib

print("\nSaving best model and preprocessing objects...")
joblib.dump(best_nb_model, 'ecg_naive_bayes_best_model.pkl')
joblib.dump(scaler, 'ecg_naive_bayes_scaler.pkl')
joblib.dump(imputer, 'ecg_naive_bayes_imputer.pkl')
print("Model and preprocessing objects saved successfully.")

# 12. Print all GridSearch results
print("\nAll GridSearch Results:")
cv_results = pd.DataFrame(grid_search.cv_results_)
results_columns = ['param_var_smoothing', 'param_priors', 'mean_test_score', 'std_test_score', 'rank_test_score']
print(cv_results[results_columns].sort_values('rank_test_score').head(10))