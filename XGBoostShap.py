import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import shap


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

# 6. Scale features using StandardScaler (needed for XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"Training data shape: {X_train_scaled.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation data shape: {X_val_scaled.shape}")
print(f"Validation labels shape: {y_val.shape}")
print(f"Test data shape: {X_test_scaled.shape}")
print(f"Test labels shape: {y_test.shape}")

# Load the saved model
try:
    print("Attempting to load saved model...")
    # Try loading model from the native XGBoost format
    best_xgb_model = XGBClassifier()
    best_xgb_model.load_model('ecg_xgb_model.json')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Saved model not found. Please train the model first using XGBoost.py")
    exit(1)

# SHAP Analysis
print("\nPerforming SHAP analysis...")

# Use a subset of the training data for SHAP analysis
n_samples_for_shap = min(500, X_train.shape[0])
X_shap_sample = X_train.iloc[:n_samples_for_shap]
X_shap_sample_scaled = X_train_scaled[:n_samples_for_shap]

# Create a SHAP explainer object for the XGBoost model
# For XGBoost, we need to use TreeExplainer
explainer = shap.TreeExplainer(best_xgb_model)

# Calculate SHAP values
# Note: For XGBoost models, SHAP values are usually calculated on scaled data
shap_values = explainer.shap_values(X_shap_sample_scaled)

# For binary classification, XGBoost might return a single array instead of a list
# Check if shap_values is a list or a single array
if isinstance(shap_values, list):
    print("Binary classification detected. Using positive class SHAP values.")
    # If it's a list, take the positive class (usually index 1)
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

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
top_features_df.to_csv('xgb_shap_top_features.csv', index=False)
print("Top features saved to 'xgb_shap_top_features.csv'")

# 1. Create a custom SHAP Feature Importance Bar Plot
plt.figure(figsize=(12, 10))
y_pos = np.arange(len(top_features))
plt.barh(y_pos, top_feature_values, color='#0099ff')
plt.yticks(y_pos, top_feature_names, fontsize=12)
plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=12)
plt.title('XGBoost SHAP Feature Importance', fontsize=14)
plt.tight_layout()
plt.savefig('xgb_shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("SHAP feature importance plot saved as 'xgb_shap_feature_importance.png'")

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
plt.title('XGBoost SHAP Summary Plot', fontsize=16)

# Improve spacing and layout
plt.tight_layout()

# Save the figure
plt.savefig('xgb_shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("SHAP summary plot saved as 'xgb_shap_summary_plot.png'")

# 3. Create a SHAP Dependence Plot for the most important feature
most_important_feature = top_feature_names[0]
most_important_idx = list(X_train.columns).index(most_important_feature)

plt.figure(figsize=(10, 8))
shap.dependence_plot(
    most_important_idx,
    shap_values,
    X_shap_sample_scaled,
    feature_names=X_train.columns,
    interaction_index=None,
    show=False
)
plt.title(f'SHAP Dependence Plot for {most_important_feature}', fontsize=14)
plt.tight_layout()
plt.savefig('xgb_shap_dependence_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"SHAP dependence plot for {most_important_feature} saved as 'xgb_shap_dependence_plot.png'")

# 4. Create a SHAP Force Plot for a single prediction
# Select a random sample from the test set
sample_idx = 0
X_sample = X_test_scaled[sample_idx:sample_idx+1]
shap_values_sample = explainer.shap_values(X_sample)
if isinstance(shap_values_sample, list):
    shap_values_sample = shap_values_sample[1] if len(shap_values_sample) > 1 else shap_values_sample[0]

# Create the force plot and save it
plt.figure(figsize=(20, 3))
force_plot = shap.force_plot(
    explainer.expected_value if hasattr(explainer, 'expected_value') else explainer.expected_value[1],
    shap_values_sample[0],
    X_test.iloc[sample_idx],
    feature_names=list(X_train.columns),
    matplotlib=True,
    show=False
)
plt.title(f'SHAP Force Plot for Sample Patient {test_patient_ids.iloc[sample_idx]}', fontsize=14)
plt.tight_layout()
plt.savefig('xgb_shap_force_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"SHAP force plot saved as 'xgb_shap_force_plot.png'")

print("\nSHAP analysis completed successfully!")