import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB

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
    print("Loading Naive Bayes model...")
    nb_model = joblib.load('ecg_naive_bayes_model.pkl')
except FileNotFoundError:
    print("Model not found. Retraining Naive Bayes model...")
    # Train a new model using the same parameters as in your original code
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    # Save the model for future use
    joblib.dump(nb_model, 'ecg_naive_bayes_model.pkl')

print(f"Model loaded: {nb_model}")

# Convert back to DataFrames for better visualization
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_val_df = pd.DataFrame(X_val_scaled, columns=feature_names)

# 3. SHAP Analysis
print("\nPerforming SHAP analysis...")

# Create a SHAP explainer for the Naive Bayes model
# For Naive Bayes, we'll use the KernelExplainer which can work with any model
# We take a smaller sample of the training data to speed up the computation
sample_size = min(100, len(X_train_df))
background = X_train_df.sample(sample_size, random_state=42)


# Create the explainer with prediction function that returns class 1 probability only
def model_predict_proba_class1(X):
    return nb_model.predict_proba(X)[:, 1]


# Create the explainer
explainer = shap.KernelExplainer(model_predict_proba_class1, background)

# Calculate SHAP values for a subset of validation set for efficiency
# You can increase this if you have the computational resources
n_samples = min(100, len(X_val_df))
val_sample = X_val_df.sample(n_samples, random_state=42)
val_indices = val_sample.index

print(f"Computing SHAP values for {n_samples} validation samples...")
# With this approach we only get SHAP values for class 1 probability
shap_values = explainer.shap_values(val_sample)

# 4. SHAP Visualizations
print("\nGenerating SHAP visualizations...")

# Set up a larger figure size for better readability
plt.figure(figsize=(12, 8))

# Summary plot showing the distribution of SHAP values for each feature
print("\nCreating SHAP summary plot...")
# Create proper Explanation object for newer SHAP versions
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=np.repeat(explainer.expected_value, len(val_sample)),
    data=val_sample.values,
    feature_names=feature_names
)

shap.summary_plot(shap_explanation, show=False, plot_size=(12, 8), max_display=20)
plt.tight_layout()
plt.savefig('nb_shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar plot showing mean absolute SHAP values (feature importance)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_explanation, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig('nb_shap_feature_importance.png', dpi=300, bbox_inches='tight')
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

# 6. Compare SHAP feature importance with Naive Bayes feature importance
print("\nComparing SHAP importance with Naive Bayes internal feature importance...")

# For GaussianNB, calculate feature importance as the difference in means
# between classes, scaled by variance
class_0_means = nb_model.theta_[0]  # Mean for class 0
class_1_means = nb_model.theta_[1]  # Mean for class 1
class_variances = nb_model.var_  # Variance for each feature

# Calculate absolute difference in means, scaled by variance
mean_diffs = np.abs(class_1_means - class_0_means)
importance = mean_diffs / np.sqrt(np.mean(class_variances, axis=0))

# Create DataFrame for NB feature importance
nb_importance = pd.DataFrame({
    'Feature': feature_names,
    'NB Importance': importance,
    'Mean |SHAP|': [mean_abs_shap.get(feature, 0) for feature in feature_names]
})
nb_importance = nb_importance.sort_values('NB Importance', ascending=False)

# Print top 20 features by NB importance
print("\nTop 20 features by Naive Bayes importance:")
print(nb_importance.head(20))

# Calculate Spearman rank correlation between NB importance and SHAP values
from scipy.stats import spearmanr

rank_correlation, p_value = spearmanr(nb_importance['NB Importance'], nb_importance['Mean |SHAP|'])
print(f"\nSpearman rank correlation between NB importance and SHAP values: {rank_correlation:.4f} (p={p_value:.4e})")

# Create a correlation plot
plt.figure(figsize=(10, 8))
plt.scatter(nb_importance['NB Importance'], nb_importance['Mean |SHAP|'], alpha=0.6)
plt.xlabel('Naive Bayes Importance')
plt.ylabel('Mean |SHAP| Value')
plt.title('Correlation between Naive Bayes Importance and SHAP Values')

# Add feature labels to the top points
top_n = 10
for i in range(top_n):
    feature = nb_importance['Feature'].iloc[i]
    x = nb_importance['NB Importance'].iloc[i]
    y = nb_importance['Mean |SHAP|'].iloc[i]
    plt.annotate(feature, (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

plt.tight_layout()
plt.savefig('nb_importance_vs_shap.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Analyze individual predictions
print("\nSHAP explanations for individual examples:")

# Select a few examples from validation set
num_examples = 5
np.random.seed(42)  # For reproducibility
example_indices = val_sample.index[:min(num_examples, len(val_sample))]

for i, idx in enumerate(example_indices):
    # Find position in the val_sample DataFrame
    val_sample_idx = val_sample.index.get_indexer([idx])[0]

    # Get SHAP values for this example
    individual_shap = shap_values[val_sample_idx]

    # Get prediction information
    actual_idx = y_val.index.get_indexer([idx])[0] if idx in y_val.index else -1
    actual_label = y_val.iloc[actual_idx] if actual_idx >= 0 else "Unknown"

    # Find the original index in X_val_scaled
    orig_idx = X_val.index.get_indexer([idx])[0] if idx in X_val.index else -1
    if orig_idx >= 0:
        pred_prob = nb_model.predict_proba(X_val_scaled[orig_idx:orig_idx + 1])[0, 1]
        pred_label = 1 if pred_prob >= 0.5 else 0
    else:
        pred_prob = nb_model.predict_proba(val_sample.iloc[[val_sample_idx]].values)[0, 1]
        pred_label = 1 if pred_prob >= 0.5 else 0

    # Print information about this example
    print(f"\nExample #{i + 1} (Index: {idx}):")
    print(f"Actual class: {actual_label}")
    print(f"Predicted class: {pred_label}")
    print(f"Prediction probability: {pred_prob:.4f}")

    # Create waterfall plot for this example
    plt.figure(figsize=(12, 8))

    # Create an Explanation object
    example_explanation = shap.Explanation(
        values=individual_shap,
        base_values=explainer.expected_value,
        data=val_sample.iloc[val_sample_idx].values,
        feature_names=feature_names
    )

    shap.waterfall_plot(example_explanation, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(f'nb_shap_waterfall_example_{i + 1}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Get top contributing features for this example
    example_values = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': individual_shap
    })
    example_values = example_values.sort_values('SHAP Value', key=abs, ascending=False)
    print("Top 10 features influencing this prediction:")
    print(example_values.head(10))

# 8. Save SHAP values for further analysis
print("\nSaving SHAP analysis results...")
np.save('nb_shap_values_val_set.npy', shap_values)
shap_importance.to_csv('nb_shap_feature_importance.csv', index=False)
nb_importance.to_csv('nb_internal_vs_shap_importance.csv', index=False)

print(
    "\nSHAP analysis for Naive Bayes complete! Check the generated PNG files and CSV files for visualizations and detailed results.")