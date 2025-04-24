import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import shap
import joblib

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

# Load the saved model or train a new one
try:
    print("Attempting to load saved model...")
    best_dt_model = joblib.load('ecg_decision_tree_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Saved model not found. Training a new model...")
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

    # Save the trained model
    joblib.dump(best_dt_model, 'ecg_decision_tree_model.pkl')
    print("Model saved successfully!")

# SHAP Analysis
print("\nPerforming SHAP analysis...")

# Use a subset of the training data for SHAP analysis
n_samples_for_shap = min(500, X_train.shape[0])
X_shap_sample = X_train.iloc[:n_samples_for_shap]

# Create a SHAP explainer object for the decision tree model
explainer = shap.TreeExplainer(best_dt_model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_shap_sample)

# For binary classification, we want the positive class values
if isinstance(shap_values, list):
    print("Binary classification detected. Using positive class SHAP values.")
    shap_values = shap_values[1]

# Calculate the mean absolute SHAP value for each feature
feature_importance = []
for i, name in enumerate(X_train.columns):
    # Calculate mean absolute value for this feature
    feature_importance.append((name, np.abs(shap_values[:, i]).mean()))

# Sort features by importance
feature_importance.sort(key=lambda x: x[1], reverse=True)

# Get top 20 features
top_features = feature_importance[:20]
top_feature_names = [feature[0] for feature in top_features]
top_feature_values = [feature[1] for feature in top_features]

# Create a DataFrame with top features and save it
top_features_df = pd.DataFrame({
    'Feature': top_feature_names,
    'SHAP Importance': top_feature_values
})
print("\nTop 20 features by SHAP importance:")
print(top_features_df)

# Save the DataFrame to a CSV file
top_features_df.to_csv('dt_shap_top_features.csv', index=False)
print("Top features saved to 'dt_shap_top_features.csv'")

# 1. Create a custom SHAP Feature Importance Bar Plot
plt.figure(figsize=(12, 10))
y_pos = np.arange(len(top_features))
plt.barh(y_pos, top_feature_values, color='#0099ff')
plt.yticks(y_pos, top_feature_names, fontsize=12)
plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=12)
plt.title('Decision Tree SHAP Feature Importance', fontsize=14)
plt.tight_layout()
plt.savefig('dt_shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("SHAP feature importance plot saved as 'dt_shap_feature_importance.png'")

# 2. Create SHAP Summary Plot (Dot) with improved styling
plt.figure(figsize=(14, 10))

# Configure plot appearance to match your example
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Get indices of the top 20 features by importance
top_feature_indices = [list(X_train.columns).index(feat[0]) for feat in top_features]

# Filter SHAP values to only include top features
shap_values_top = shap_values[:, top_feature_indices]

# Create a DataFrame with just the top features to ensure feature names are preserved
X_top_df = X_shap_sample.iloc[:, top_feature_indices].copy()
X_top_df.columns = top_feature_names  # Explicitly set column names

# Create the summary plot with custom parameters
shap.summary_plot(
    shap_values_top,
    X_top_df,  # Use the DataFrame with named columns
    plot_type="dot",
    plot_size=(14, 10),
    color_bar_label="Feature value",
    show=False,
    # Additional parameters to customize appearance
    alpha=0.8,  # Make dots slightly transparent
    max_display=20,  # Show top 20 features
    cmap="cool"  # Use a color map similar to your example (blue to pink)
)

# Add a vertical line at x=0
plt.axvline(x=0, color='purple', linestyle='-', alpha=0.6)

# Adjust the x-label
plt.xlabel('SHAP value (impact on model output)', fontsize=14)

# Add a title
plt.title('Decision Tree SHAP Summary Plot', fontsize=16)

# Improve spacing and layout
plt.tight_layout()

# Save the figure
plt.savefig('dt_shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("SHAP summary plot saved as 'dt_shap_summary_plot.png'")


print("\nSHAP analysis completed successfully!")