import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

# 1. Load the data and model
print("Loading data...")
# Load the extracted features from CSV files
train_features = pd.read_csv('ecg_cardiac_features_train.csv')
val_features = pd.read_csv('ecg_cardiac_features_val.csv')
test_features = pd.read_csv('ecg_cardiac_features_test.csv')

# Load labels
y_train = pd.read_csv('Y_train_1.csv', index_col=0)
y_val = pd.read_csv('Y_valid_1.csv', index_col=0)
y_test = pd.read_csv('Y_test_1.csv', index_col=0)

# For binary classification, choose the second column as the positive class
y_train = y_train.iloc[:, 1]
y_val = y_val.iloc[:, 1]
y_test = y_test.iloc[:, 1]

# Extract patient IDs and prepare feature matrices
train_patient_ids = train_features['patient_id']
val_patient_ids = val_features['patient_id']
test_patient_ids = test_features['patient_id']

# Get common columns across all datasets (exclude patient_id)
common_cols = list(set(train_features.columns) &
                   set(val_features.columns) &
                   set(test_features.columns))
common_cols = [col for col in common_cols if col != 'patient_id']

# Use only common columns
X_train = train_features[common_cols]
X_val = val_features[common_cols]
X_test = test_features[common_cols]

# Store feature names for SHAP analysis
feature_names = X_train.columns.tolist()

# Impute missing values
print("Preprocessing data...")
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 2. Load the trained KNN model (or retrain if needed)
try:
    print("Loading tuned KNN model...")
    best_knn_model = joblib.load('ecg_knn_model.pkl')
except FileNotFoundError:
    print("KNN model not found. Retraining model...")
    # Train a basic KNN model
    best_knn_model = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    )
    best_knn_model.fit(X_train_scaled, y_train)

print(f"Model loaded: {best_knn_model}")

# 3. SHAP Analysis
print("\nPerforming SHAP analysis...")

# Convert back to DataFrames for better visualization
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_val_df = pd.DataFrame(X_val_scaled, columns=feature_names)

# For KNN, we'll use KernelExplainer since there's no model-specific explainer
# Use a smaller background dataset for computational efficiency
# Create a smaller representative subset of the training data
np.random.seed(42)  # For reproducibility
background_size = min(100, len(X_train_df))
background_indices = np.random.choice(len(X_train_df), background_size, replace=False)
background = X_train_df.iloc[background_indices]


# Define a predict function that returns the probability of the positive class
def predict_proba_positive(x):
    return best_knn_model.predict_proba(x)[:, 1]


# Create a SHAP explainer with the Kernel explainer
print("Creating KernelExplainer - this may take some time...")
explainer = shap.KernelExplainer(predict_proba_positive, background)

# Calculate SHAP values for a subset of the validation set (for speed)
# Adjust this number based on your computational resources
sample_size = min(100, len(X_val_df))  # Reduced from 200 to 100 for faster computation
print(f"Calculating SHAP values for {sample_size} validation samples...")
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(len(X_val_df), sample_size, replace=False)
X_val_sample = X_val_df.iloc[sample_indices]
y_val_sample = y_val.iloc[sample_indices]

# Calculate SHAP values
shap_values = explainer.shap_values(X_val_sample)

# 4. SHAP Visualizations
print("\nGenerating SHAP visualizations...")

# Set up a larger figure size for better readability
plt.figure(figsize=(12, 8))

# Summary plot showing the distribution of SHAP values for each feature
print("\nCreating SHAP summary plot...")
shap.summary_plot(shap_values, X_val_sample, show=False, max_display=20)
plt.tight_layout()
plt.savefig('knn_shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar plot showing mean absolute SHAP values (feature importance)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_val_sample, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig('knn_shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Feature Analysis using SHAP values
print("\nAnalyzing feature contributions...")

# Create a DataFrame for easier manipulation of SHAP values
shap_df = pd.DataFrame(shap_values, columns=feature_names)

# Calculate mean absolute SHAP values for each feature
mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=False)

# Create DataFrame for feature importance based on SHAP
shap_importance = pd.DataFrame({
    'Feature': mean_abs_shap.index,
    'Mean |SHAP|': mean_abs_shap.values
})

# Print the top 20 most important features according to SHAP
print("\nTop 20 most important features by SHAP value:")
print(shap_importance.head(20))

# 6. Analyze individual predictions
print("\nSHAP explanations for individual examples:")

# Select a few examples from the sample
num_examples = 5
example_indices = np.random.choice(len(X_val_sample), num_examples, replace=False)

for i, idx in enumerate(example_indices):
    # Get the original index in the validation set
    original_idx = sample_indices[idx]

    # Get SHAP values for this example
    individual_shap = shap_values[idx]

    # Get prediction information
    actual_label = y_val_sample.iloc[idx]
    pred_prob = best_knn_model.predict_proba(X_val_sample.iloc[idx:idx + 1])[0, 1]
    pred_label = 1 if pred_prob >= 0.5 else 0

    # Print information about this example
    print(f"\nExample #{i + 1} (Original Index: {original_idx}):")
    print(f"Patient ID: {val_patient_ids.iloc[original_idx]}")
    print(f"Actual class: {actual_label}")
    print(f"Predicted class: {pred_label}")
    print(f"Prediction probability: {pred_prob:.4f}")

    # Create force plot for this example (waterfall plots can be challenging with KernelExplainer)
    plt.figure(figsize=(12, 8))
    shap_explanation = shap.Explanation(
        values=individual_shap,
        base_values=explainer.expected_value,
        data=X_val_sample.iloc[idx].values,
        feature_names=feature_names
    )
    shap.plots.force(shap_explanation, show=False, matplotlib=True)
    plt.tight_layout()
    plt.savefig(f'knn_shap_force_example_{i + 1}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Get top contributing features for this example
    example_values = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': individual_shap
    })
    example_values = example_values.sort_values('SHAP Value', key=abs, ascending=False)
    print("Top 10 features influencing this prediction:")
    print(example_values.head(10))

# 7. Save SHAP values for further analysis
print("\nSaving SHAP analysis results...")
np.save('knn_shap_values_val_set.npy', shap_values)
shap_importance.to_csv('knn_shap_feature_importance.csv', index=False)

# 8. Dependence plots for top features
print("\nCreating dependence plots for top features...")
top_features = shap_importance['Feature'].head(3).tolist()  # Limit to top 3 for time efficiency

for feature in top_features:
    feature_idx = feature_names.index(feature)
    plt.figure(figsize=(12, 8))
    shap.dependence_plot(feature_idx, shap_values, X_val_sample, show=False)
    plt.tight_layout()
    plt.savefig(f'knn_shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\nSHAP analysis complete! Check the generated PNG files and CSV files for visualizations and detailed results.")