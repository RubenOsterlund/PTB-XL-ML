import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

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

# Drop non-feature columns
X_train = train_features.drop('patient_id', axis=1)
X_val = val_features.drop('patient_id', axis=1)
X_test = test_features.drop('patient_id', axis=1)

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

# 2. Load the trained model (or retrain if needed)
try:
    print("Loading tuned SVM model...")
    best_svm_model = joblib.load('ecg_svm_model.pkl')
except FileNotFoundError:
    print("SVM model not found. Retraining model...")
    # Train a new SVM model with default parameters
    best_svm_model = SVC(
        probability=True,
        class_weight='balanced',
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42
    )
    best_svm_model.fit(X_train_scaled, y_train)

    # Save the model for future use
    joblib.dump(best_svm_model, 'ecg_svm_model.pkl')

print(f"Model loaded: {best_svm_model}")

# 3. SHAP Analysis
print("\nPerforming SHAP analysis...")

# Convert back to DataFrames for better visualization
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_val_df = pd.DataFrame(X_val_scaled, columns=feature_names)

# Use kmeans to summarize the background data - address the performance warning
print("Creating summarized background data with K-means clustering...")
# Using fewer background samples (100) for better performance
background_data = shap.kmeans(X_train_df, 100)

# Using a subset of validation data for analysis to make it computationally feasible
analysis_samples = min(300, X_val_df.shape[0])
X_val_subset = X_val_df.iloc[:analysis_samples]
y_val_subset = y_val.iloc[:analysis_samples] if hasattr(y_val, 'iloc') else y_val[:analysis_samples]

print("Creating SHAP explainer...")
# Define the prediction function
predict_fn = lambda x: best_svm_model.predict_proba(x)[:, 1]

# Create the explainer with the summarized background data
explainer = shap.KernelExplainer(predict_fn, background_data)

# Calculate SHAP values for the validation subset
print(f"Calculating SHAP values for {analysis_samples} validation samples...")
shap_values = explainer.shap_values(X_val_subset)

# 4. SHAP Visualizations
print("\nGenerating SHAP visualizations...")

# Set up a larger figure size for better readability
plt.figure(figsize=(12, 8))

# Summary plot showing the distribution of SHAP values for each feature
print("\nCreating SHAP summary plot...")
shap.summary_plot(shap_values, X_val_subset, feature_names=feature_names,
                  show=False, plot_size=(12, 8), max_display=20)
plt.tight_layout()
plt.savefig('svm_shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar plot showing mean absolute SHAP values (feature importance)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_val_subset, feature_names=feature_names,
                  plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig('svm_shap_feature_importance.png', dpi=300, bbox_inches='tight')
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

# 6. Compare SHAP feature importance with model coefficients (if linear kernel)
print("\nComparing SHAP importance with model parameters...")

if best_svm_model.kernel == 'linear':
    coefficients = best_svm_model.coef_[0]
    coef_importance = np.abs(coefficients)

    # Create DataFrame for coefficient importance
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient Magnitude': coef_importance,
        'Mean |SHAP|': [mean_abs_shap.get(feature, 0) for feature in feature_names]
    })
    coef_df = coef_df.sort_values('Coefficient Magnitude', ascending=False)

    # Print top 20 features by coefficient magnitude
    print("\nTop 20 features by coefficient magnitude:")
    print(coef_df.head(20))

    # Calculate Spearman rank correlation between coefficient magnitudes and SHAP values
    from scipy.stats import spearmanr

    rank_correlation, p_value = spearmanr(coef_df['Coefficient Magnitude'], coef_df['Mean |SHAP|'])
    print(
        f"\nSpearman rank correlation between coefficient magnitudes and SHAP values: {rank_correlation:.4f} (p={p_value:.4e})")

    # Save coefficient comparison
    coef_df.to_csv('svm_coefficient_vs_shap_importance.csv', index=False)
else:
    print(f"Direct coefficient analysis not available for '{best_svm_model.kernel}' kernel.")
    print("SHAP values provide model-agnostic feature importance instead.")

# 7. Analyze individual predictions
print("\nSHAP explanations for individual examples:")

# Select a few examples from validation subset
num_examples = 5
np.random.seed(42)  # For reproducibility
example_indices = np.random.choice(len(X_val_subset), num_examples, replace=False)

for i, idx in enumerate(example_indices):
    actual_idx = idx  # This is the index in the subset

    # Get prediction information
    actual_label = y_val_subset.iloc[actual_idx] if hasattr(y_val_subset, 'iloc') else y_val_subset[actual_idx]
    pred_prob = best_svm_model.predict_proba(X_val_subset.iloc[[actual_idx]])[0, 1]
    pred_label = 1 if pred_prob >= 0.5 else 0

    # Get patient ID (adjust index if using a subset)
    patient_id = val_patient_ids.iloc[actual_idx]

    # Print information about this example
    print(f"\nExample #{i + 1} (Index: {actual_idx}):")
    print(f"Patient ID: {patient_id}")
    print(f"Actual class: {actual_label}")
    print(f"Predicted class: {pred_label}")
    print(f"Prediction probability: {pred_prob:.4f}")

    # Create force plot for this example
    plt.figure(figsize=(12, 8))
    shap.force_plot(
        explainer.expected_value,
        shap_values[actual_idx],
        X_val_subset.iloc[actual_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'svm_shap_force_example_{i + 1}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Get top contributing features for this example
    example_values = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values[actual_idx]
    })
    example_values = example_values.sort_values('SHAP Value', key=abs, ascending=False)
    print("Top 10 features influencing this prediction:")
    print(example_values.head(10))

# Create dependence plots for top features
print("\nCreating dependence plots for top features...")
top_features = shap_importance['Feature'].iloc[:3].tolist()  # Top 3 features
for feature in top_features:
    plt.figure(figsize=(12, 8))
    shap.dependence_plot(
        feature,
        shap_values,
        X_val_subset,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'svm_shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. Save SHAP values for further analysis
print("\nSaving SHAP analysis results...")
np.save('svm_shap_values_val_set.npy', shap_values)
shap_importance.to_csv('svm_shap_feature_importance.csv', index=False)


print(
    "\nSVM SHAP analysis complete! Check the generated PNG files and CSV files for visualizations and detailed results.")