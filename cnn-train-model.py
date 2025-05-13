import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load and prepare data
df = pd.read_csv('newdataset.csv')
print(df.head())

X = df.iloc[:,1:].values  # Convert to numpy array
y = df.iloc[:,0].values

# Get number of classes and unique class values
unique_classes = np.unique(y)
num_classes = len(unique_classes)
print(f"Number of classes: {num_classes}")
print(f"Classes: {unique_classes}")

# Create string labels for reports and plotting
string_labels = [f"Class {int(cls)}" if np.issubdtype(type(cls), np.number) else str(cls) 
                for cls in unique_classes]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for CNN (samples, timesteps, features)
# Treat each frame as having 33 timesteps (landmarks) with 2 features (x,y)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 33, 2)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_onehot, test_size=0.3, random_state=42, stratify=y_encoded
)

# Build the CNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(33, 2)),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred, target_names=string_labels))

# Generate confusion matrix
cm = confusion_matrix(y_test_classes, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
ax = plt.subplot()
sn.set(font_scale=1.4)
sn.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 16})
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title("CNN Model Confusion Matrix")
ax.xaxis.set_ticklabels(string_labels)
ax.yaxis.set_ticklabels(string_labels)
plt.tight_layout()
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Save the model
model.save('exercise_cnn_model.h5')
print("Model saved as 'exercise_cnn_model.h5'")

# Also save the scaler for preprocessing new data
with open('cnn_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'cnn_scaler.pkl'")