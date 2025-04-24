import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import tree

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

# 7. Create and train Decision Tree model with default parameters
dt_model = DecisionTreeClassifier(
    class_weight='balanced',  # Help with class imbalance
    random_state=42  # For reproducibility
)

# 8. Train the model
print("Training initial Decision Tree model...")
dt_model.fit(X_train, y_train)

# 9. Evaluate on validation set
val_predictions = dt_model.predict(X_val)
val_proba = dt_model.predict_proba(X_val)[:, 1]  # For ROC-AUC

print("\nInitial Validation Performance:")
print(confusion_matrix(y_val, val_predictions))
print(classification_report(y_val, val_predictions))
print(f"ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")

# 10. Hyperparameter tuning with Grid Search
print("\nPerforming grid search for hyperparameter tuning...")
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',  # Use ROC-AUC for scoring
    n_jobs=-1  # Use all available processors
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
best_dt_model = grid_search.best_estimator_

# 11. Evaluate tuned model on validation set
tuned_val_predictions = best_dt_model.predict(X_val)
tuned_val_proba = best_dt_model.predict_proba(X_val)[:, 1]

print("\nTuned Model Validation Performance:")
print(confusion_matrix(y_val, tuned_val_predictions))
print(classification_report(y_val, tuned_val_predictions))
print(f"ROC-AUC: {roc_auc_score(y_val, tuned_val_proba):.4f}")


# 12. Final evaluation on test set
test_predictions = best_dt_model.predict(X_test)
test_proba = best_dt_model.predict_proba(X_test)[:, 1]

print("\nTest Performance:")
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print(f"ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")

# 13. Feature importance for Decision Tree model
feature_names = X_train.columns
feature_importances = best_dt_model.feature_importances_

feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nTop 20 most important features:")
print(feature_importance.head(20))

# 14. Visualize the decision tree (may be complex for many features)
# Limit visualization to max_depth=3 for clarity if the tree is large
plt.figure(figsize=(20, 10))
tree.plot_tree(best_dt_model, max_depth=3, feature_names=list(feature_names),
               filled=True, rounded=True, fontsize=10)
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
print("Decision tree visualization saved as 'decision_tree_visualization.png'")


# 15. Save the trained model and preprocessing objects
import joblib

print("\nSaving model and preprocessing objects...")
joblib.dump(best_dt_model, 'ecg_decision_tree_model.pkl')
print("Model and preprocessing objects saved successfully.")