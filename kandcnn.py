import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import os
from tensorflow.keras.layers import Input, GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from google.colab import drive

# Mount Google Drive with proper error handling
try:
    drive.mount('/content/drive')
    print("Google Drive successfully mounted")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    print("Please run this cell again and complete the authentication process")

# Set path to your uploaded data files (update this path as needed)
data_path = '/content/drive/MyDrive/PTB_XL_Data/'

# Load the preprocessed data
# Select which diagnostic category to use (1, 2, or 3)
# 1: MI (Myocardial Infarction) vs NORM (Normal)
# 2: STTC (ST/T Change) vs NORM
# 3: CD (Conduction Disturbance) vs NORM
category = 1  # Change this to 2 or 3 to use different diagnostic categories

X_train = np.load(data_path + f'X_train_{category}.npy')
X_valid = np.load(data_path + f'X_valid_{category}.npy')
X_test = np.load(data_path + f'X_test_{category}.npy')

Y_train = pd.read_csv(data_path + f'Y_train_{category}.csv', index_col=0)
Y_valid = pd.read_csv(data_path + f'Y_valid_{category}.csv', index_col=0)
Y_test = pd.read_csv(data_path + f'Y_test_{category}.csv', index_col=0)

# Convert Y dataframes to numpy arrays for training
classes = Y_train.columns.tolist()
y_train = Y_train.values
y_valid = Y_valid.values
y_test = Y_test.values

print(f"Training with category {category}: {classes}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Data preprocessing
# Normalize the data (scale to range [0,1])
def normalize_data(data):
    # Normalize each lead separately
    for i in range(data.shape[0]):
        for lead in range(data.shape[2]):
            lead_data = data[i, :, lead]
            min_val = np.min(lead_data)
            max_val = np.max(lead_data)
            if max_val > min_val:  # Avoid division by zero
                data[i, :, lead] = (lead_data - min_val) / (max_val - min_val)
    return data

X_train = normalize_data(X_train)
X_valid = normalize_data(X_valid)
X_test = normalize_data(X_test)

# Data visualization
def plot_sample_ecg(X, y, classes, index=0):
    plt.figure(figsize=(15, 8))

    n_leads = X.shape[2]
    for i in range(n_leads):
        plt.subplot(3, 4, i+1)
        plt.plot(X[index, :, i])
        plt.title(f'Lead {i+1}')
        plt.grid(True)

    # Get the class label
    class_idx = np.argmax(y[index])
    class_name = classes[class_idx]

    plt.suptitle(f'ECG Sample - Class: {class_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Plot a sample from each class
for class_idx in range(len(classes)):
    class_samples = np.where(np.argmax(y_train, axis=1) == class_idx)[0]
    if len(class_samples) > 0:
        plot_sample_ecg(X_train, y_train, classes, class_samples[0])

def residual_block(inputs, filters, kernel_size=3):
    x = inputs
    shortcut = inputs

    # First Conv Layer
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Second Conv Layer
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # If input and output dimensions don't match, adjust shortcut
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add shortcut to output
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x

def create_cnn_model(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # First block with residual connection
    x = Conv1D(32, 5, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = residual_block(x, 32)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    # Second block with residual connection
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    # Third block with residual connection
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    # Global pooling
    x1 = GlobalMaxPooling1D()(x)  # Max features
    x2 = GlobalAveragePooling1D()(x)  # Average features
    x = Concatenate()([x1, x2])  # Combine different feature views

    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create and compile the model
input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
num_classes = len(classes)

model = create_cnn_model(input_shape, num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Visualize the model architecture
#plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Set up callbacks
checkpoint_path = "ecg_model_best.keras"
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    reduce_lr
]

# Train the model and evaluate on validation set
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

# Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Generate classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

# Generate ROC curve and AUC score
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()

# Save the model
model.save(data_path + 'ecg_cnn_model.keras')
print(f"Model saved to {data_path + 'ecg_cnn_model.keras'}")
