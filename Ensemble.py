import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Load your data using the same pattern as in your LogisticReg.py
print("Loading data...")
test_features = pd.read_csv('ecg_cardiac_features_test.csv')
y_test = pd.read_csv('Y_test_1.csv', index_col=0)
y_test = y_test.iloc[:, 1]  # Take the second column for binary classification

# If you need validation data for ensemble fitting
val_features = pd.read_csv('ecg_cardiac_features_val.csv')
y_val = pd.read_csv('Y_valid_1.csv', index_col=0)
y_val = y_val.iloc[:, 1]

# Store patient IDs for reference
test_patient_ids = test_features['patient_id']
val_patient_ids = val_features['patient_id']

# Store column names for later use with tree-based models
feature_names = test_features.columns.drop('patient_id').tolist()

# Drop non-feature columns
X_test = test_features.drop('patient_id', axis=1)
X_val = val_features.drop('patient_id', axis=1)

# Load the preprocessing objects
print("Loading preprocessing objects...")
try:
    imputer = joblib.load('ecg_feature_imputer.pkl')
    scaler = joblib.load('ecg_feature_scaler.pkl')

    # Apply imputation to all data
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # Convert back to DataFrame to preserve feature names for tree-based models
    X_val_imputed_df = pd.DataFrame(X_val_imputed, columns=feature_names)
    X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=feature_names)

    # Apply scaling only to models that need it
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    print("Preprocessing objects loaded and applied successfully.")
except FileNotFoundError:
    print("Preprocessing objects not found. Creating new imputer and scaler...")
    # If you haven't saved these, create new ones
    imputer = SimpleImputer(strategy='mean')
    X_val_imputed = imputer.fit_transform(X_val)
    X_test_imputed = imputer.transform(X_test)

    # Convert back to DataFrame to preserve feature names for tree-based models
    X_val_imputed_df = pd.DataFrame(X_val_imputed, columns=feature_names)
    X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=feature_names)

    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

# Create DMatrix objects for XGBoost
dtest = xgb.DMatrix(X_test_imputed)
dval = xgb.DMatrix(X_val_imputed)

# Define paths to your saved models - REMOVED Gaussian Process
model_paths = {
    'XGBoost': 'ecg_xgb_model.json',
    'Random Forest': 'ecg_random_forest_model.pkl',
    'Decision Tree': 'ecg_decision_tree_model.pkl',
    'SVM': 'ecg_svm_model.pkl',
    'Logistic Regression': 'ecg_lr_model_tuned.pkl',
    'Naive Bayes': 'ecg_naive_bayes_model.pkl',
    'KNN': 'ecg_knn_model.pkl'
}

# Define which models need scaling
models_needing_scaling = ['SVM', 'Logistic Regression', 'Naive Bayes', 'KNN']
models_not_needing_scaling = ['Random Forest', 'Decision Tree']
special_models = ['XGBoost']  # Models with special handling

# Load all available models more defensively
print("Loading saved models...")
loaded_models = {}
for name, path in model_paths.items():
    try:
        # For XGBoost models specifically
        if name == 'XGBoost' and path.endswith('.json'):
            model = xgb.Booster()
            model.load_model(path)
        else:
            model = joblib.load(path)
        loaded_models[name] = model
        print(f"Successfully loaded {name} model")
    except Exception as e:
        print(f"Warning: Could not load {name} model from '{path}': {str(e)}")
        continue


# PreprocessingWrapper class
class PreprocessingWrapper:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.is_xgboost = name == 'XGBoost'
        self.needs_scaling = name in models_needing_scaling
        self.needs_feature_names = name in models_not_needing_scaling

    def predict(self, X):
        # Special case for XGBoost
        if self.is_xgboost:
            if isinstance(X, np.ndarray) and np.array_equal(X, X_test_scaled):
                preds = self.model.predict(dtest)
            elif isinstance(X, np.ndarray) and np.array_equal(X, X_val_scaled):
                preds = self.model.predict(dval)
            else:
                # If we can't match the input exactly, create a new DMatrix
                preds = self.model.predict(xgb.DMatrix(X))

            # Convert probabilities to binary predictions - threshold 0.5
            return (preds > 0.5).astype(int)

        # For models that need scaling
        elif self.needs_scaling:
            return self.model.predict(X)

        # For models that need feature names (tree-based models)
        elif self.needs_feature_names:
            if isinstance(X, np.ndarray) and np.array_equal(X, X_test_scaled):
                return self.model.predict(X_test_imputed_df)
            elif isinstance(X, np.ndarray) and np.array_equal(X, X_val_scaled):
                return self.model.predict(X_val_imputed_df)
            else:
                # If we can't match the input data, convert to DataFrame with feature names
                return self.model.predict(pd.DataFrame(X, columns=feature_names))
        else:
            # For other models
            return self.model.predict(X)

    def predict_proba(self, X):
        # For XGBoost which doesn't have predict_proba
        if self.is_xgboost:
            # Get raw predictions
            if isinstance(X, np.ndarray) and np.array_equal(X, X_test_scaled):
                preds = self.model.predict(dtest)
            elif isinstance(X, np.ndarray) and np.array_equal(X, X_val_scaled):
                preds = self.model.predict(dval)
            else:
                preds = self.model.predict(xgb.DMatrix(X))

            # Convert to sklearn-compatible format [prob_class_0, prob_class_1]
            # Make sure we have shape (n_samples, 2)
            return np.vstack((1 - preds, preds)).T

        # For models that need scaling
        elif self.needs_scaling:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # If model doesn't have predict_proba, try to fake it based on predict
                try:
                    preds = self.model.predict(X)
                    proba = np.zeros((len(X), 2))
                    proba[np.arange(len(X)), preds.astype(int)] = 1
                    return proba
                except:
                    raise AttributeError(f"Model {self.name} doesn't have predict_proba method")

        # For models that need feature names (tree-based models)
        elif self.needs_feature_names:
            if not hasattr(self.model, 'predict_proba'):
                # Try to fake it with predict
                try:
                    if isinstance(X, np.ndarray) and np.array_equal(X, X_test_scaled):
                        preds = self.model.predict(X_test_imputed_df)
                    elif isinstance(X, np.ndarray) and np.array_equal(X, X_val_scaled):
                        preds = self.model.predict(X_val_imputed_df)
                    else:
                        preds = self.model.predict(pd.DataFrame(X, columns=feature_names))

                    proba = np.zeros((len(X), 2))
                    proba[np.arange(len(X)), preds.astype(int)] = 1
                    return proba
                except:
                    raise AttributeError(f"Model {self.name} doesn't have predict_proba method")

            if isinstance(X, np.ndarray) and np.array_equal(X, X_test_scaled):
                return self.model.predict_proba(X_test_imputed_df)
            elif isinstance(X, np.ndarray) and np.array_equal(X, X_val_scaled):
                return self.model.predict_proba(X_val_imputed_df)
            else:
                # If we can't match the input data, convert to DataFrame with feature names
                return self.model.predict_proba(pd.DataFrame(X, columns=feature_names))

        # For other models
        else:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # Try to fake it with predict
                try:
                    preds = self.model.predict(X)
                    proba = np.zeros((len(X), 2))
                    proba[np.arange(len(X)), preds.astype(int)] = 1
                    return proba
                except:
                    raise AttributeError(f"Model {self.name} doesn't have predict_proba method")

    def fit(self, X, y):
        # XGBoost Booster doesn't have a fit method once loaded
        if self.is_xgboost:
            # Silently return self without warning
            return self

        # For models that need scaling
        elif self.needs_scaling:
            if hasattr(self.model, 'fit'):
                try:
                    return self.model.fit(X, y)
                except:
                    # Some models might fail to fit, just return self
                    return self
            else:
                return self

        # For models that need feature names (tree-based models)
        elif self.needs_feature_names:
            if hasattr(self.model, 'fit'):
                try:
                    if isinstance(X, np.ndarray) and np.array_equal(X, X_val_scaled):
                        return self.model.fit(X_val_imputed_df, y)
                    else:
                        # If we can't match the input data, convert to DataFrame with feature names
                        return self.model.fit(pd.DataFrame(X, columns=feature_names), y)
                except:
                    # Some models might fail to fit, just return self
                    return self
            else:
                return self

        # For other models
        else:
            if hasattr(self.model, 'fit'):
                try:
                    return self.model.fit(X, y)
                except:
                    # Some models might fail to fit, just return self
                    return self
            else:
                return self


# Wrap models with appropriate preprocessing
wrapped_models = {}
for name, model in loaded_models.items():
    wrapped_models[name] = PreprocessingWrapper(model, name)

# Evaluate individual models on the test set with appropriate preprocessing
print("\n--- Individual Model Performance ---")
results = {}
model_predictions = {}
model_probabilities = {}


# Add utility function to verify predictions contain both classes
def check_predictions_quality(name, preds):
    if len(np.unique(preds)) < 2:
        print(f"WARNING: {name} model is only predicting one class: {np.unique(preds)}")
        return False
    class_counts = np.bincount(preds.astype(int))
    print(f"{name} class distribution: {class_counts}")
    if min(class_counts) < 5:  # Arbitrary threshold to detect severe imbalance
        print(f"WARNING: {name} model has very few predictions for one class!")
    return True


for name, wrapped_model in wrapped_models.items():
    try:
        # Get predictions using the wrapper (which handles preprocessing differences)
        y_pred = wrapped_model.predict(X_test_scaled)
        model_predictions[name] = y_pred

        # Check prediction quality
        valid_predictions = check_predictions_quality(name, y_pred)
        if not valid_predictions:
            print(f"Skipping {name} model due to poor predictions")
            continue

        # Get probabilities if available (needed for soft voting)
        try:
            y_proba = wrapped_model.predict_proba(X_test_scaled)
            # Ensure probabilities are in the right format
            if y_proba.shape[1] == 2:
                model_probabilities[name] = y_proba[:, 1]  # Probability of class 1
                auc = roc_auc_score(y_test, y_proba[:, 1])
                print(f"{name}: ROC-AUC = {auc:.4f}")
            else:
                print(f"WARNING: {name} returned invalid probability shape: {y_proba.shape}")
        except Exception as proba_error:
            print(f"{name} doesn't support predict_proba, will use hard voting only: {str(proba_error)}")

        # Calculate accuracy
        accuracy = (y_pred == y_test).mean()
        results[name] = accuracy
        print(f"{name}: Accuracy = {accuracy:.4f}")

        # Print classification report
        print(classification_report(y_test, y_pred, zero_division=0))
    except Exception as e:
        print(f"Error evaluating {name} model: {str(e)}")

# Sort models by performance
sorted_models = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\nModels ranked by accuracy:")
for name, acc in sorted_models:
    print(f"{name}: {acc:.4f}")

# Create a subset of your data for fitting the ensemble
# Using validation data is preferable as test data should remain unseen
X_ensemble_fit = X_val_scaled
y_ensemble_fit = y_val

# Keep only models that have valid predictions and support predict_proba for soft voting
soft_voting_models = {name: model for name, model in wrapped_models.items()
                      if name in model_probabilities and name in results}

# Filter out models that only predict one class
valid_models = {name: model for name, model in wrapped_models.items()
                if name in results and len(np.unique(model_predictions[name])) > 1}

print(f"\nNumber of models with valid predictions: {len(valid_models)}")
print(f"Number of models supporting soft voting: {len(soft_voting_models)}")


# Create a custom VotingClassifier wrapper to handle our custom models
class CustomVotingClassifier:
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.fitted = False

    def fit(self, X, y):
        for _, estimator in self.estimators:
            estimator.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Classifier not fitted")

        # Get predictions from all estimators
        predictions = []
        for name, estimator in self.estimators:
            preds = estimator.predict(X)
            print(f"Model {name} predictions shape: {preds.shape}, unique values: {np.unique(preds)}")
            predictions.append(preds)

        if self.voting == 'hard':
            # For hard voting, take the most common prediction for each sample
            # Convert predictions to a matrix of shape (n_samples, n_estimators)
            predictions = np.array(predictions).T  # shape (n_samples, n_estimators)
            print(f"Combined predictions shape: {predictions.shape}")

            # Apply weights if specified
            if self.weights is not None:
                # Using bincount with weights on each row
                maj = np.apply_along_axis(
                    lambda x: np.bincount(x, weights=self.weights, minlength=2).argmax(),
                    axis=1, arr=predictions.astype(int)
                )
            else:
                # Using mode on each row
                maj = np.apply_along_axis(
                    lambda x: np.bincount(x, minlength=2).argmax(),
                    axis=1, arr=predictions.astype(int)
                )

            print(f"Hard voting results - Class distribution: {np.bincount(maj.astype(int))}")
            return maj
        else:  # soft voting
            # Get probabilities from all estimators
            probas = []
            for name, estimator in self.estimators:
                proba = estimator.predict_proba(X)
                print(f"Model {name} probabilities shape: {proba.shape}")
                probas.append(proba)

            # Average probabilities (weighted if specified)
            if self.weights is not None:
                avg_proba = np.zeros_like(probas[0])
                for i, proba in enumerate(probas):
                    avg_proba += self.weights[i] * proba
                avg_proba /= sum(self.weights)
            else:
                avg_proba = np.mean(probas, axis=0)

            print(f"Average probabilities shape: {avg_proba.shape}")

            # Return class with highest probability
            predictions = np.argmax(avg_proba, axis=1)
            print(f"Soft voting results - Class distribution: {np.bincount(predictions.astype(int))}")
            return predictions

    def predict_proba(self, X):
        if not self.fitted or self.voting == 'hard':
            raise ValueError("Predict proba only available for fitted soft voting classifiers")

        # Get probabilities from all estimators
        probas = []
        for _, estimator in self.estimators:
            probas.append(estimator.predict_proba(X))

        # Average probabilities (weighted if specified)
        if self.weights is not None:
            avg_proba = np.zeros_like(probas[0])
            for i, proba in enumerate(probas):
                avg_proba += self.weights[i] * proba
            avg_proba /= sum(self.weights)
        else:
            avg_proba = np.mean(probas, axis=0)

        return avg_proba


# APPROACH 1: Hard Voting with valid models
print("\n--- Building Hard Voting Ensemble ---")
estimators = [(name, model) for name, model in valid_models.items()]

if len(estimators) >= 2:  # Need at least 2 models for voting
    hard_voting_clf = CustomVotingClassifier(estimators=estimators, voting='hard')

    try:
        # Fit the voting classifier
        hard_voting_clf.fit(X_ensemble_fit, y_ensemble_fit)

        # Predict with the hard voting ensemble
        y_pred_hard = hard_voting_clf.predict(X_test_scaled)
        acc_hard = (y_pred_hard == y_test).mean()
        print(f"Hard Voting Ensemble Accuracy: {acc_hard:.4f}")
        print(classification_report(y_test, y_pred_hard))
    except Exception as e:
        print(f"Error with hard voting ensemble: {str(e)}")
else:
    print("Not enough valid models for hard voting ensemble")

# APPROACH 2: Soft Voting (with models that support predict_proba)
if len(soft_voting_models) >= 2:  # Need at least 2 models for voting
    print("\n--- Building Soft Voting Ensemble ---")
    soft_estimators = [(name, model) for name, model in soft_voting_models.items()]
    soft_voting_clf = CustomVotingClassifier(estimators=soft_estimators, voting='soft')

    try:
        # Fit the soft voting classifier
        soft_voting_clf.fit(X_ensemble_fit, y_ensemble_fit)

        # Predict with the soft voting ensemble
        y_pred_soft = soft_voting_clf.predict(X_test_scaled)
        y_proba_soft = soft_voting_clf.predict_proba(X_test_scaled)[:, 1]
        acc_soft = (y_pred_soft == y_test).mean()
        auc_soft = roc_auc_score(y_test, y_proba_soft)

        print(f"Soft Voting Ensemble Accuracy: {acc_soft:.4f}")
        print(f"Soft Voting Ensemble ROC-AUC: {auc_soft:.4f}")
        print(classification_report(y_test, y_pred_soft))
    except Exception as e:
        print(f"Error with soft voting ensemble: {str(e)}")
else:
    print("Not enough models support predict_proba for soft voting")

# APPROACH 3: Weighted Soft Voting based on individual performance
if len(soft_voting_models) >= 2:
    print("\n--- Building Weighted Soft Voting Ensemble ---")
    # Use only models that support predict_proba and in soft_voting_models
    weight_models = {name: model for name, model in soft_voting_models.items() if name in results}
    weight_estimators = [(name, model) for name, model in weight_models.items()]

    # Calculate weights based on accuracy
    weights = [results[name] for name, _ in weight_models.items()]

    # Normalize weights to sum to 1
    weights = np.array(weights) / np.sum(weights)

    print("Models and weights:")
    for (name, _), weight in zip(weight_estimators, weights):
        print(f"{name}: {weight:.4f}")

    weighted_voting_clf = CustomVotingClassifier(
        estimators=weight_estimators,
        voting='soft',
        weights=weights
    )

    try:
        # Fit the weighted voting classifier
        weighted_voting_clf.fit(X_ensemble_fit, y_ensemble_fit)

        # Predict with the weighted voting ensemble
        y_pred_weighted = weighted_voting_clf.predict(X_test_scaled)
        y_proba_weighted = weighted_voting_clf.predict_proba(X_test_scaled)[:, 1]
        acc_weighted = (y_pred_weighted == y_test).mean()
        auc_weighted = roc_auc_score(y_test, y_proba_weighted)

        print(f"Weighted Soft Voting Ensemble Accuracy: {acc_weighted:.4f}")
        print(f"Weighted Soft Voting Ensemble ROC-AUC: {auc_weighted:.4f}")
        print(classification_report(y_test, y_pred_weighted))
    except Exception as e:
        print(f"Error with weighted soft voting ensemble: {str(e)}")

# APPROACH 4: Custom Stacking using our wrappers (since sklearn's StackingClassifier might not work with our custom wrappers)
if len(soft_voting_models) >= 2:
    print("\n--- Building Custom Stacking Ensemble ---")


    class CustomStackingClassifier:
        def __init__(self, estimators, final_estimator):
            self.estimators = estimators
            self.final_estimator = final_estimator
            self.fitted = False

        def fit(self, X, y):
            # Split validation data for meta-model training
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Get predictions from base models using cross-validation
            meta_features = np.zeros((X.shape[0], len(self.estimators)))

            for i, (name, estimator) in enumerate(self.estimators):
                print(f"Training base model: {name}")
                # Use cross-validation to get out-of-fold predictions
                fold_indices = list(kf.split(X))
                for train_idx, val_idx in fold_indices:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]

                    # Train the estimator
                    estimator.fit(X_train, y_train)

                    # Get probabilities for the positive class
                    try:
                        proba = estimator.predict_proba(X_val)
                        if proba.shape[1] == 2:
                            meta_features[val_idx, i] = proba[:, 1]
                        else:
                            meta_features[val_idx, i] = estimator.predict(X_val)
                    except:
                        # If predict_proba is not available, use predict
                        meta_features[val_idx, i] = estimator.predict(X_val)

                # Retrain on the full dataset
                estimator.fit(X, y)

            # Check class distribution in meta-features
            meta_preds = (meta_features > 0.5).astype(int)
            for i, (name, _) in enumerate(self.estimators):
                print(f"Model {name} meta-predictions distribution: {np.bincount(meta_preds[:, i])}")

            print(f"Meta-features shape: {meta_features.shape}")
            # Train the meta-classifier on the meta-features
            print("Training meta-classifier")
            self.final_estimator.fit(meta_features, y)
            self.fitted = True
            return self

        def predict(self, X):
            if not self.fitted:
                raise ValueError("Classifier not fitted")

            # Get predictions from all base models
            meta_features = np.zeros((X.shape[0], len(self.estimators)))
            for i, (name, estimator) in enumerate(self.estimators):
                try:
                    proba = estimator.predict_proba(X)
                    if proba.shape[1] == 2:
                        meta_features[:, i] = proba[:, 1]
                    else:
                        meta_features[:, i] = estimator.predict(X)
                except:
                    meta_features[:, i] = estimator.predict(X)

            # Make final prediction using the meta-classifier
            return self.final_estimator.predict(meta_features)

        def predict_proba(self, X):
            if not self.fitted:
                raise ValueError("Classifier not fitted")

            # Get predictions from all base models
            meta_features = np.zeros((X.shape[0], len(self.estimators)))
            for i, (name, estimator) in enumerate(self.estimators):
                try:
                    proba = estimator.predict_proba(X)
                    if proba.shape[1] == 2:
                        meta_features[:, i] = proba[:, 1]
                    else:
                        meta_features[:, i] = estimator.predict(X)
                except:
                    meta_features[:, i] = estimator.predict(X)

            # Make final prediction using the meta-classifier
            return self.final_estimator.predict_proba(meta_features)


    # Use LogisticRegression as the meta-classifier
    meta_clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    stack_estimators = [(name, model) for name, model in soft_voting_models.items()]

    stacking_clf = CustomStackingClassifier(
        estimators=stack_estimators,
        final_estimator=meta_clf
    )

    try:
        # Fit the stacking classifier
        stacking_clf.fit(X_ensemble_fit, y_ensemble_fit)

        # Predict with the stacking ensemble
        y_pred_stack = stacking_clf.predict(X_test_scaled)
        y_proba_stack = stacking_clf.predict_proba(X_test_scaled)[:, 1]
        acc_stack = (y_pred_stack == y_test).mean()
        auc_stack = roc_auc_score(y_test, y_proba_stack)

        print(f"Stacking Ensemble Accuracy: {acc_stack:.4f}")
        print(f"Stacking Ensemble ROC-AUC: {auc_stack:.4f}")
        print(classification_report(y_test, y_pred_stack))
    except Exception as e:
        print(f"Error with stacking ensemble: {str(e)}")
else:
    print("Not enough models support predict_proba for stacking")

# APPROACH 5: Hybrid ensemble (combining the best results from different approaches)
print("\n--- Building Hybrid Ensemble ---")

# Create a meta-ensemble that combines all our previous ensemble methods
try:
    # Check which ensemble methods we successfully created
    ensembles = {}
    ensemble_weights = {}

    if 'hard_voting_clf' in locals() and 'acc_hard' in locals():
        ensembles['Hard Voting'] = hard_voting_clf
        ensemble_weights['Hard Voting'] = acc_hard

    if 'soft_voting_clf' in locals() and 'acc_soft' in locals():
        ensembles['Soft Voting'] = soft_voting_clf
        ensemble_weights['Soft Voting'] = acc_soft

    if 'weighted_voting_clf' in locals() and 'acc_weighted' in locals():
        ensembles['Weighted Voting'] = weighted_voting_clf
        ensemble_weights['Weighted Voting'] = acc_weighted

    if 'stacking_clf' in locals() and 'acc_stack' in locals():
        ensembles['Stacking'] = stacking_clf
        ensemble_weights['Stacking'] = acc_stack

    # Include the best individual model too
    best_model_name, best_acc = sorted_models[0]
    ensembles['Best Model'] = wrapped_models[best_model_name]
    ensemble_weights['Best Model'] = best_acc

    print(f"Number of ensemble methods available: {len(ensembles)}")
    print("Ensemble methods and their accuracies:")
    for name, acc in ensemble_weights.items():
        print(f"{name}: {acc:.4f}")

    if len(ensembles) >= 2:
        # Create a final meta-ensemble using majority voting
        print("Creating final hybrid ensemble using majority voting")

        # Function to get majority vote from all ensembles
        def hybrid_predict(X):
            all_predictions = []
            for name, model in ensembles.items():
                try:
                    preds = model.predict(X)
                    all_predictions.append(preds)
                    print(f"Hybrid: {name} predictions shape: {preds.shape}")
                except Exception as e:
                    print(f"Error getting predictions from {name}: {str(e)}")

            # Stack predictions and take majority vote
            if all_predictions:
                stacked_preds = np.column_stack(all_predictions)
                print(f"Stacked predictions shape: {stacked_preds.shape}")

                # Weighted voting based on accuracy
                ensemble_names = list(ensembles.keys())
                weights = [ensemble_weights[name] for name in ensemble_names]
                weights = np.array(weights) / np.sum(weights)

                # Apply weighted voting
                final_preds = np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int), weights=[weights[i] for i in range(len(weights)) if i < len(x)], minlength=2).argmax(),
                    axis=1, arr=stacked_preds
                )
                return final_preds
            else:
                return np.zeros(X.shape[0])  # Fallback

        # Function to get average probabilities from all ensembles
        def hybrid_predict_proba(X):
            all_probas = []
            for name, model in ensembles.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        all_probas.append(proba)
                        print(f"Hybrid: {name} probabilities shape: {proba.shape}")
                except Exception as e:
                    print(f"Error getting probabilities from {name}: {str(e)}")

            # Average the probabilities
            if all_probas:
                # Get ensemble names that successfully provided probabilities
                ensemble_names = list(ensembles.keys())[:len(all_probas)]
                weights = [ensemble_weights[name] for name in ensemble_names]
                weights = np.array(weights) / np.sum(weights)

                # Apply weighted average
                avg_proba = np.zeros_like(all_probas[0])
                for i, proba in enumerate(all_probas):
                    avg_proba += weights[i] * proba

                return avg_proba
            else:
                # Fallback - return [0, 1] for all positive class
                proba = np.zeros((X.shape[0], 2))
                proba[:, 1] = 1
                return proba

        # Make predictions with the hybrid ensemble
        y_pred_hybrid = hybrid_predict(X_test_scaled)
        y_proba_hybrid = hybrid_predict_proba(X_test_scaled)[:, 1]

        # Calculate performance metrics
        acc_hybrid = (y_pred_hybrid == y_test).mean()
        auc_hybrid = roc_auc_score(y_test, y_proba_hybrid)

        print(f"Hybrid Ensemble Accuracy: {acc_hybrid:.4f}")
        print(f"Hybrid Ensemble ROC-AUC: {auc_hybrid:.4f}")
        print(classification_report(y_test, y_pred_hybrid))
    else:
        print("Not enough ensemble methods for hybrid ensemble")
except Exception as e:
    print(f"Error creating hybrid ensemble: {str(e)}")

# Summary of all ensemble methods
print("\n--- Ensemble Methods Summary ---")
ensemble_results = {}

# Individual best model
best_model_name, best_acc = sorted_models[0]
ensemble_results["Best Individual Model"] = best_acc

# Add results from all ensemble methods
if 'acc_hard' in locals():
    ensemble_results["Hard Voting"] = acc_hard
if 'acc_soft' in locals():
    ensemble_results["Soft Voting"] = acc_soft
if 'acc_weighted' in locals():
    ensemble_results["Weighted Soft Voting"] = acc_weighted
'''
if 'acc_top' in locals():
    ensemble_results["Top-K Soft Voting"] = acc_top
'''
if 'acc_stack' in locals():
    ensemble_results["Stacking"] = acc_stack
if 'acc_hybrid' in locals():
    ensemble_results["Hybrid Ensemble"] = acc_hybrid
    '''
if 'acc_bma' in locals():
    ensemble_results["Bayesian Model Averaging"] = acc_bma
'''
# Sort results by performance
sorted_ensemble_results = sorted(ensemble_results.items(), key=lambda x: x[1], reverse=True)

print("Ensemble methods ranked by accuracy:")
for name, acc in sorted_ensemble_results:
    print(f"{name}: {acc:.4f}")

# Save the best ensemble model
try:
    best_ensemble_name, _ = sorted_ensemble_results[0]
    print(f"\nBest ensemble method: {best_ensemble_name}")

    if best_ensemble_name == "Hard Voting" and 'hard_voting_clf' in locals():
        joblib.dump(hard_voting_clf, 'ecg_hard_voting_ensemble.pkl')
        print("Saved hard voting ensemble")
    elif best_ensemble_name == "Soft Voting" and 'soft_voting_clf' in locals():
        joblib.dump(soft_voting_clf, 'ecg_soft_voting_ensemble.pkl')
        print("Saved soft voting ensemble")
    elif best_ensemble_name == "Weighted Soft Voting" and 'weighted_voting_clf' in locals():
        joblib.dump(weighted_voting_clf, 'ecg_weighted_voting_ensemble.pkl')
        print("Saved weighted soft voting ensemble")
    elif best_ensemble_name == "Stacking" and 'stacking_clf' in locals():
        joblib.dump(stacking_clf, 'ecg_stacking_ensemble.pkl')
        print("Saved stacking ensemble")
    else:
        print("Could not save best ensemble model - not available in local variables")
except Exception as e:
    print(f"Error saving best ensemble model: {str(e)}")

print("\nEnsemble evaluation complete!")



