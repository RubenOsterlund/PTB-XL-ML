import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the SVM model and preprocessing objects
svm_model = joblib.load('ecg_svm_model.pkl')

# 2. Load the CNN model
cnn_model = tf.keras.models.load_model('ecg_cnn_model.keras')


# 3. Function to load and preprocess data for both models
def load_and_preprocess_data(category=1):
    # For CNN: Load the npy files
    X_train_cnn = np.load(f'X_train_{category}.npy')
    X_valid_cnn = np.load(f'X_valid_{category}.npy')
    X_test_cnn = np.load(f'X_test_{category}.npy')

    # Normalize CNN data
    def normalize_data(data):
        for i in range(data.shape[0]):
            for lead in range(data.shape[2]):
                lead_data = data[i, :, lead]
                min_val = np.min(lead_data)
                max_val = np.max(lead_data)
                if max_val > min_val:
                    data[i, :, lead] = (lead_data - min_val) / (max_val - min_val)
        return data

    X_test_cnn = normalize_data(X_test_cnn)
    X_val_cnn = normalize_data(X_valid_cnn)

    # For SVM: Load extracted features
    X_test_features = pd.read_csv('ecg_cardiac_features_test.csv')
    X_val_features = pd.read_csv('ecg_cardiac_features_val.csv')

    # Extract patient IDs for reference
    test_patient_ids = X_test_features['patient_id']
    val_patient_ids = X_val_features['patient_id']

    # Preprocess for SVM
    X_test_svm = X_test_features.drop('patient_id', axis=1)
    X_val_svm = X_val_features.drop('patient_id', axis=1)

    # Load the imputer and scaler objects
    imputer = joblib.load('ecg_svm_imputer.pkl')
    scaler = joblib.load('ecg_svm_scaler.pkl')

    # Apply imputation and scaling
    X_val_svm_imputed = imputer.transform(X_val_svm)
    X_test_svm_imputed = imputer.transform(X_test_svm)

    X_val_svm_scaled = scaler.transform(X_val_svm_imputed)
    X_test_svm_scaled = scaler.transform(X_test_svm_imputed)

    # Load labels
    Y_valid = pd.read_csv(f'Y_valid_{category}.csv', index_col=0)
    Y_test = pd.read_csv(f'Y_test_{category}.csv', index_col=0)

    # For binary classification - convert to numpy arrays
    y_val = Y_valid.values
    y_test = Y_test.values

    return X_val_cnn, X_test_cnn, X_val_svm_scaled, X_test_svm_scaled, y_val, y_test, Y_valid.columns.tolist()


# 4. Load the data
X_val_cnn, X_test_cnn, X_val_svm, X_test_svm, y_val, y_test, class_names = load_and_preprocess_data(category=1)

# 5. Generate predictions from base models on validation set
print("Generating base model predictions...")
# SVM predictions
svm_val_proba = svm_model.predict_proba(X_val_svm)
# CNN predictions
cnn_val_proba = cnn_model.predict(X_val_cnn)

# 6. Create meta-features for training the meta-learner
# Combine probabilities from both models
meta_features_val = np.hstack([svm_val_proba, cnn_val_proba])

# 7. Train a meta-learner (LogisticRegression) on these predictions
print("Training meta-learner...")
meta_learner = LogisticRegression(max_iter=1000)
# For binary classification, we need the class labels
y_val_labels = np.argmax(y_val, axis=1) if y_val.shape[1] > 1 else y_val
meta_learner.fit(meta_features_val, y_val_labels)

# 8. Generate predictions from base models on test set
svm_test_proba = svm_model.predict_proba(X_test_svm)
cnn_test_proba = cnn_model.predict(X_test_cnn)

# 9. Create meta-features for test set
meta_features_test = np.hstack([svm_test_proba, cnn_test_proba])

# 10. Generate final ensemble predictions
ensemble_proba = meta_learner.predict_proba(meta_features_test)
ensemble_pred = meta_learner.predict(meta_features_test)

# For comparison, get individual model predictions
svm_pred = np.argmax(svm_test_proba, axis=1)
cnn_pred = np.argmax(cnn_test_proba, axis=1)
y_test_labels = np.argmax(y_test, axis=1) if y_test.shape[1] > 1 else y_test


# 11. Evaluate all models
def evaluate_model(y_true, y_pred, y_proba, model_name):
    print(f"\n{model_name} Performance:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=class_names))

    # For binary classification
    if len(class_names) == 2:
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        print(f"ROC-AUC: {roc_auc:.4f}")
    else:
        # For multiclass, use one-vs-rest AUC
        y_true_one_hot = np.eye(len(class_names))[y_true]
        roc_auc = roc_auc_score(y_true_one_hot, y_proba, multi_class='ovr', average='macro')
        print(f"ROC-AUC (One-vs-Rest): {roc_auc:.4f}")
    return roc_auc


# Evaluate each model
print("\nModel Evaluation:")
svm_auc = evaluate_model(y_test_labels, svm_pred, svm_test_proba, "SVM")
cnn_auc = evaluate_model(y_test_labels, cnn_pred, cnn_test_proba, "CNN")
ensemble_auc = evaluate_model(y_test_labels, ensemble_pred, ensemble_proba, "Ensemble")

# 12. Visualize results
plt.figure(figsize=(12, 8))

# Confusion matrices
plt.subplot(2, 2, 1)
sns.heatmap(confusion_matrix(y_test_labels, svm_pred), annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(2, 2, 2)
sns.heatmap(confusion_matrix(y_test_labels, cnn_pred), annot=True, fmt='d', cmap='Blues')
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(2, 2, 3)
sns.heatmap(confusion_matrix(y_test_labels, ensemble_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Ensemble Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# ROC Comparison
plt.subplot(2, 2, 4)
models = ['SVM', 'CNN', 'Ensemble']
aucs = [svm_auc, cnn_auc, ensemble_auc]
plt.bar(models, aucs)
plt.title('ROC-AUC Comparison')
plt.ylabel('AUC Score')
plt.ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig('ensemble_results.png')
plt.show()

# 13. Save the ensemble model
ensemble_model = {
    'svm_model': svm_model,
    'cnn_model': cnn_model,
    'meta_learner': meta_learner
}
joblib.dump(ensemble_model, 'ecg_ensemble_model.pkl')
print("Ensemble model saved to 'ecg_ensemble_model.pkl'")