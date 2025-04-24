import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

# 4. Get common columns across all datasets
common_cols = list(set(train_features.columns) &
                   set(val_features.columns) &
                   set(test_features.columns))
common_cols = [col for col in common_cols if col != 'patient_id']

# Use only common columns
X_train = train_features[common_cols]
X_val = val_features[common_cols]
X_test = test_features[common_cols]

# 5. Impute missing values (NaN)
# NaN values can occur in feature extraction when certain features couldn't be reliably measured
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

# 7. Create and train KNN model with default parameters
knn_model = KNeighborsClassifier(
    n_neighbors=5,  # Default value, will tune this
    weights='uniform',  # Default value, will tune this
    metric='minkowski',  # Euclidean distance by default
    n_jobs=-1  # Use all available processors
)

# 8. Train the model
print("Training initial KNN model...")
knn_model.fit(X_train_scaled, y_train)

# 9. Evaluate on validation set
val_predictions = knn_model.predict(X_val_scaled)
val_proba = knn_model.predict_proba(X_val_scaled)[:, 1]  # For ROC-AUC

print("\nInitial Validation Performance:")
print(confusion_matrix(y_val, val_predictions))
print(classification_report(y_val, val_predictions))
print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

# 10. Hyperparameter tuning with Grid Search
print("\nPerforming grid search for hyperparameter tuning...")
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
}

grid_search = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    param_grid,
    cv=5,
    scoring='roc_auc',  # Use ROC-AUC for scoring
    n_jobs=-1  # Use all available processors
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
best_knn_model = grid_search.best_estimator_

# 11. Evaluate tuned model on validation set
tuned_val_predictions = best_knn_model.predict(X_val_scaled)
tuned_val_proba = best_knn_model.predict_proba(X_val_scaled)[:, 1]

print("\nTuned Model Validation Performance:")
print(confusion_matrix(y_val, tuned_val_predictions))
print(classification_report(y_val, tuned_val_predictions))
print(f"ROC-AUC: {roc_auc_score(y_val, tuned_val_proba):.4f}")


# 12. Final evaluation on test set
test_predictions = best_knn_model.predict(X_test_scaled)
test_proba = best_knn_model.predict_proba(X_test_scaled)[:, 1]

print("\nTest Performance:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print(f"ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")

# 13. Save the trained model and preprocessing objects
import joblib

print("\nSaving model and preprocessing objects...")
joblib.dump(best_knn_model, 'ecg_knn_model.pkl')
print("Model and preprocessing objects saved successfully.")

# 14. Compare KNN with different distances and k values
print("\nComparing different KNN configurations:")
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
k_values = [3, 5, 11, 21]

results = []

for metric in distance_metrics:
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric=metric, n_jobs=-1)
        knn.fit(X_train_scaled, y_train)
        val_predictions = knn.predict(X_val_scaled)
        val_proba = knn.predict_proba(X_val_scaled)[:, 1]
        roc_auc = roc_auc_score(y_val, val_proba)

        results.append({
            'metric': metric,
            'k': k,
            'roc_auc': roc_auc
        })

        print(f"Metric: {metric}, k={k}, ROC-AUC: {roc_auc:.4f}")

# Create a DataFrame to easily view the results
results_df = pd.DataFrame(results)
print("\nResults summary:")
print(results_df.sort_values('roc_auc', ascending=False))