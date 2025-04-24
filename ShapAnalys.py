import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# 1. Load the seven dataframes with feature importance
model_files = [
    'lr_shap_feature_importance.csv',  # Logistic Regression
    'rf_shap_top_features.csv',  # Random Forest
    'dt_shap_top_features.csv',  # Decision Tree
    'xgb_shap_top_features.csv',  # XGBoost
    'svm_shap_feature_importance.csv',  # SVM
    'nb_shap_feature_importance.csv',  # Naive Bayes
    'knn_shap_feature_importance.csv'  # K-Nearest Neighbors
]

model_names = [
    'Logistic Regression',
    'Random Forest',
    'Decision Tree',
    'XGBoost',
    'SVM',
    'Naive Bayes',
    'KNN'
]

# Dictionary to store each model's dataframe
model_dfs = {}

# Load each model's feature importance dataframe
print("Loading feature importance dataframes...")
for i, (file, model_name) in enumerate(zip(model_files, model_names)):
    try:
        df = pd.read_csv(file)
        print(f"Loaded {model_name} features from {file}")
        model_dfs[model_name] = df
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping {model_name}.")

if not model_dfs:
    raise Exception("No model data could be loaded. Please check file paths.")

# 2. Analyze feature frequency across models
print("\nAnalyzing feature frequency across models...")

# Collect all unique features
all_features = set()
for model, df in model_dfs.items():
    feature_col = 'Feature' if 'Feature' in df.columns else df.columns[0]
    all_features.update(df[feature_col].tolist())

print(f"Total unique features across all models: {len(all_features)}")

# Count feature occurrences across models
feature_counts = Counter()
feature_ranks = {}
feature_importance_values = {}

for model, df in model_dfs.items():
    feature_col = 'Feature' if 'Feature' in df.columns else df.columns[0]
    importance_col = 'SHAP Importance' if 'SHAP Importance' in df.columns else df.columns[1]

    # Count occurrences of each feature
    for feature in df[feature_col]:
        feature_counts[feature] += 1

    # Store feature ranks for each model (position in the list)
    for i, feature in enumerate(df[feature_col]):
        if feature not in feature_ranks:
            feature_ranks[feature] = {}
        feature_ranks[feature][model] = i + 1  # Add 1 to make it 1-indexed

    # Store feature importance values
    for i, row in df.iterrows():
        feature = row[feature_col]
        importance = row[importance_col]
        if feature not in feature_importance_values:
            feature_importance_values[feature] = {}
        feature_importance_values[feature][model] = importance

# 3. Get top 10 most frequently occurring features
top_features = [feature for feature, count in feature_counts.most_common(10)]
print("\nTop 10 most common features across models:")
for i, (feature, count) in enumerate(feature_counts.most_common(10)):
    print(f"{i + 1}. {feature} (appears in {count} models)")

# 4. Create a comparison dataframe for the top 10 features
comparison_data = []

for feature in top_features:
    row = {'Feature': feature, 'Occurrence': feature_counts[feature]}

    # Add rank and importance for each model
    for model in model_names:
        if model in model_dfs:
            # Add rank if feature present in this model
            if feature in feature_ranks and model in feature_ranks[feature]:
                row[f'{model} Rank'] = feature_ranks[feature][model]
            else:
                row[f'{model} Rank'] = None

            # Add importance value if feature present in this model
            if feature in feature_importance_values and model in feature_importance_values[feature]:
                row[f'{model} Importance'] = feature_importance_values[feature][model]
            else:
                row[f'{model} Importance'] = 0

    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

# Calculate average rank and importance across models where the feature appears
for i, row in comparison_df.iterrows():
    feature = row['Feature']

    # Calculate average rank
    ranks_list = []
    for model in model_names:
        if f'{model} Rank' in row.index and row[f'{model} Rank'] is not None:
            ranks_list.append(row[f'{model} Rank'])

    comparison_df.at[i, 'Average Rank'] = sum(ranks_list) / len(ranks_list) if ranks_list else None

    # Calculate average importance
    importances = []
    for model in model_names:
        if f'{model} Importance' in row.index and row[f'{model} Importance'] > 0:
            importances.append(row[f'{model} Importance'])

    comparison_df.at[i, 'Average Importance'] = sum(importances) / len(importances) if importances else 0

# Sort by occurrence first, then by average importance
comparison_df = comparison_df.sort_values(by=['Occurrence', 'Average Importance'], ascending=[False, False])

# Save the comparison dataframe
comparison_df.to_csv('model_feature_comparison.csv', index=False)
print("\nFeature comparison dataframe saved to 'model_feature_comparison.csv'")

# 5. Create visualizations

# Set up the plotting style
plt.style.use('ggplot')
sns.set(font_scale=1.1)
colors = sns.color_palette("viridis", len(model_names))

# 5.1 Feature occurrence bar chart
plt.figure(figsize=(12, 8))
feature_occurrence = comparison_df[['Feature', 'Occurrence']].sort_values('Occurrence', ascending=False)
sns.barplot(x='Occurrence', y='Feature', data=feature_occurrence, palette='Blues_d')
plt.title('Number of Models Where Feature Appears in Top 20', fontsize=16)
plt.xlabel('Number of Models', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig('feature_occurrence_across_models.png', dpi=300, bbox_inches='tight')
plt.close()
print("Feature occurrence chart saved as 'feature_occurrence_across_models.png'")

# 5.2 Feature importance heatmap
plt.figure(figsize=(14, 10))

# Prepare data for heatmap
heatmap_data = pd.DataFrame(index=top_features)
for model in model_names:
    if model in model_dfs:
        model_values = []
        for feature in top_features:
            if feature in feature_importance_values and model in feature_importance_values[feature]:
                model_values.append(feature_importance_values[feature][model])
            else:
                model_values.append(0)
        heatmap_data[model] = model_values

# Convert all values to float to avoid dtype issues
for col in heatmap_data.columns:
    heatmap_data[col] = heatmap_data[col].astype(float)

# Create heatmap
ax = sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.3f', linewidths=.5)
plt.title('Feature Importance Across Models', fontsize=16)
plt.ylabel('Feature', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Feature importance heatmap saved as 'feature_importance_heatmap.png'")

# 5.3 Feature rank radar chart (for top 5 features)
top_5_features = top_features[:5]


# Create a function for radar charts
def radar_chart(features, title):
    # Number of variables
    N = len(model_names)

    # Create a figure
    fig = plt.figure(figsize=(10, 10))

    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Initialize the radar plot
    ax = fig.add_subplot(111, polar=True)

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], model_names, size=12)

    # Draw y-axis labels (ranks - inverted scale for better visualization)
    max_rank = 20  # Assume max rank is 20 (as in top 20 features)
    ax.set_rlabel_position(0)
    plt.yticks([5, 10, 15, 20], ["15", "10", "5", "1"], color="grey", size=10)
    plt.ylim(0, max_rank)

    # Plot each feature
    for i, feature in enumerate(features):
        values = []
        for model in model_names:
            # Get rank or max_rank+1 if not present
            if feature in feature_ranks and model in feature_ranks[feature]:
                # Invert rank for better visualization (1 should be at the outer edge)
                values.append(max_rank - feature_ranks[feature][model] + 1)
            else:
                values.append(0)  # Not in top 20

        # Close the loop
        values += values[:1]

        # Plot values
        ax.plot(angles, values, linewidth=2, label=feature)
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=16, y=1.1)

    return fig


# Create radar chart for top 5 features
radar_fig = radar_chart(top_5_features, 'Feature Rank Comparison Across Models')
radar_fig.savefig('feature_rank_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("Feature rank radar chart saved as 'feature_rank_radar_chart.png'")

# 6. Create a composite score for each feature
print("\nCalculating composite importance scores...")

# Normalize importance values within each model
normalized_importance = {}
for model in model_names:
    if model in model_dfs:
        feature_col = 'Feature' if 'Feature' in model_dfs[model].columns else model_dfs[model].columns[0]
        importance_col = 'SHAP Importance' if 'SHAP Importance' in model_dfs[model].columns else \
        model_dfs[model].columns[1]

        # Get max importance for normalization
        max_importance = model_dfs[model][importance_col].max()

        # Normalize each feature's importance
        for i, row in model_dfs[model].iterrows():
            feature = row[feature_col]
            importance = row[importance_col]
            if feature not in normalized_importance:
                normalized_importance[feature] = {}
            normalized_importance[feature][model] = importance / max_importance if max_importance > 0 else 0

# Calculate composite score (weighted sum of normalized importance and frequency)
composite_scores = {}
for feature in all_features:
    # Sum of normalized importance across all models
    importance_sum = sum(normalized_importance.get(feature, {}).get(model, 0) for model in model_names)
    # Composite score
    composite_scores[feature] = (importance_sum / len(model_names))

# Create and save composite score dataframe
composite_df = pd.DataFrame({
    'Feature': list(composite_scores.keys()),
    'Composite Score': list(composite_scores.values()),
    'Occurrence': [feature_counts[feature] for feature in composite_scores.keys()]
})
composite_df = composite_df.sort_values('Composite Score', ascending=False).reset_index(drop=True)

# Save top 20 by composite score
top_composite_df = composite_df.head(20)
top_composite_df.to_csv('top_features_composite_score.csv', index=False)
print("\nTop features by composite score saved to 'top_features_composite_score.csv'")

# Plot top 20 features by composite score
plt.figure(figsize=(12, 8))
sns.barplot(x='Composite Score', y='Feature', data=top_composite_df.head(10), palette='YlOrRd')
plt.title('Top 10 Features by Composite Importance Score', fontsize=16)
plt.xlabel('Composite Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig('feature_composite_score.png', dpi=300, bbox_inches='tight')
plt.close()
print("Composite score chart saved as 'feature_composite_score.png'")

# 7. Create a consolidated visualization showing model agreement
# This visualization shows which models agree on important features
plt.figure(figsize=(14, 10))

# Create a binary matrix for feature presence in each model's top 20
# And explicitly ensure all values are numeric (integers)
presence_matrix = pd.DataFrame(index=composite_df.head(20)['Feature'], columns=model_names)
for feature in presence_matrix.index:
    for model in model_names:
        if model in model_dfs:
            feature_col = 'Feature' if 'Feature' in model_dfs[model].columns else model_dfs[model].columns[0]
            # Convert to integer explicitly
            presence_matrix.loc[feature, model] = int(feature in model_dfs[model][feature_col].values)
        else:
            presence_matrix.loc[feature, model] = 0

# Convert all columns to integers to avoid dtype issues
for col in presence_matrix.columns:
    presence_matrix[col] = presence_matrix[col].astype(int)

# Use a simpler approach to create the heatmap
plt.figure(figsize=(14, 10))
plt.imshow(presence_matrix.values, cmap='YlGnBu', aspect='auto')
plt.colorbar(label='Feature Present')
plt.xticks(np.arange(len(model_names)), model_names, rotation=45, ha='right')
plt.yticks(np.arange(len(presence_matrix.index)), presence_matrix.index)
plt.title('Model Agreement on Top 20 Features', fontsize=16)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig('model_agreement_heatmap.png', dpi=300, bbox_inches='tight')

# Alternative approach using a custom matrix plotting method
plt.figure(figsize=(14, 10))
agreement_data = presence_matrix.values.astype(int)
plt.pcolormesh(agreement_data, cmap='YlGnBu', edgecolors='w', linewidth=1)
plt.colorbar(label='Feature Present')
plt.xticks(np.arange(len(model_names)) + 0.5, model_names, rotation=45, ha='right')
plt.yticks(np.arange(len(presence_matrix.index)) + 0.5, presence_matrix.index)
plt.title('Model Agreement on Top 20 Features (Alternative)', fontsize=16)
plt.ylabel('Feature', fontsize=14)
plt.tight_layout()
plt.savefig('model_agreement_heatmap_alt.png', dpi=300, bbox_inches='tight')
plt.close()
print("Model agreement heatmaps saved as 'model_agreement_heatmap.png' and 'model_agreement_heatmap_alt.png'")

print("\nAnalysis complete! Results have been saved to CSV files and visualizations.")