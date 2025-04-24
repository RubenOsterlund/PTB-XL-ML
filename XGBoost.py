import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
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

# 7. Create and train XGBoost model with initial parameters
# Note: XGBoost handles class imbalance with scale_pos_weight parameter
print("Training initial XGBoost model...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Create DMatrix objects for XGBoost (optimized data structure)
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dval = xgb.DMatrix(X_val_scaled, label=y_val)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Initial XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'auc',            # Use AUC for evaluation
    'eta': 0.1,                      # Learning rate
    'max_depth': 3,                  # Maximum tree depth
    'min_child_weight': 1,           # Minimum sum of instance weight needed in a child
    'subsample': 0.8,                # Subsample ratio of the training instances
    'colsample_bytree': 0.8,         # Subsample ratio of columns when constructing each tree
    'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
    'seed': 42                       # For reproducibility
}

# Train XGBoost model with early stopping
evallist = [(dtrain, 'train'), (dval, 'eval')]
num_round = 100  # Maximum number of boosting rounds

# Train with early stopping
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_round,
    evals=evallist,
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    verbose_eval=10            # Print evaluation every 10 rounds
)

print(f"Best score reached at iteration {xgb_model.best_iteration}")
print(f"Best score: {xgb_model.best_score}")

# 9. Evaluate on validation set
val_proba = xgb_model.predict(dval)
val_predictions = (val_proba > 0.5).astype(int)

print("\nInitial Validation Performance:")
print(confusion_matrix(y_val, val_predictions))
print(classification_report(y_val, val_predictions))
print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

# 10. Feature importance for XGBoost model
importance_scores = xgb_model.get_score(importance_type='gain')
feature_names = X_train.columns

# Create a DataFrame for better visualization
# Note: XGBoost might not use all features, so we handle missing features
importance_df = pd.DataFrame(
    {'Feature': list(importance_scores.keys()),
     'Importance': list(importance_scores.values())}
)
importance_df = importance_df.sort_values('Importance', ascending=False)

print("\nTop 20 most important features:")
print(importance_df.head(20))

# FIXED HYPERPARAMETER TUNING USING XGBOOST WITH RANDOMIZEDSEARCHCV
print("\nPerforming randomized search for hyperparameter tuning...")

# Create an XGBoost classifier for use with sklearn
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# Define parameter distributions for randomized search
param_distributions = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Use RandomizedSearchCV for faster tuning
random_search = RandomizedSearchCV(
    xgb_clf,
    param_distributions=param_distributions,
    n_iter=15,  # Try only 15 random combinations instead of all possible combinations
    cv=3,       # 3-fold cross-validation
    scoring='roc_auc',
    n_jobs=-1,  # Use all CPU cores
    random_state=42,
    verbose=1
)

# FIX: Properly fit the model without early_stopping_rounds in the fit parameters
random_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {random_search.best_params_}")
best_xgb_model = random_search.best_estimator_

# If you want to retrain with early stopping, do it after getting the best parameters
# Create a new model with the best parameters
final_xgb = xgb.XGBClassifier(
    **random_search.best_params_,
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# Fit with early stopping using eval_set
eval_set = [(X_val_scaled, y_val)]
final_xgb = xgb.XGBClassifier(
    **random_search.best_params_,
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    # Add early stopping directly in the constructor
    early_stopping_rounds=10
)

# Then fit without early_stopping parameter
final_xgb.fit(
    X_train_scaled,
    y_train,
    eval_set=eval_set,
    verbose=True
)

# Evaluate tuned model on validation set
tuned_val_proba = final_xgb.predict_proba(X_val_scaled)[:, 1]
tuned_val_predictions = (tuned_val_proba > 0.5).astype(int)

print("\nTuned Model Validation Performance:")
print(confusion_matrix(y_val, tuned_val_predictions))
print(classification_report(y_val, tuned_val_predictions))
print(f"ROC-AUC: {roc_auc_score(y_val, tuned_val_proba):.4f}")

# Final evaluation on test set
test_proba = final_xgb.predict_proba(X_test_scaled)[:, 1]
test_predictions = (test_proba > 0.5).astype(int)

print("\nTest Performance:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print(f"ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")

# Save the trained model and preprocessing objects
import joblib

print("\nSaving model and preprocessing objects...")
final_xgb.save_model('ecg_xgb_model.json')  # Native XGBoost format
print("Model and preprocessing objects saved successfully.")
