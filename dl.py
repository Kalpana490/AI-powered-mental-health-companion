# ===================================================
# LSTM + BiLSTM MODELS FOR STRESS LEVEL PREDICTION (FULL EDA)
# ===================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import joblib


# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("StressLevelDataset.csv")

print("Dataset Loaded Successfully")
print(df.head())


# ===============================
# Exploratory Data Analysis (EDA)
# ===============================

# 1. Distribution of Stress Levels
plt.figure(figsize=(6, 4))
sns.countplot(x="stress_level", data=df, palette="coolwarm")
plt.title("Distribution of Stress Levels")
plt.xlabel("Stress Level")
plt.ylabel("Count")
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="viridis")
plt.title("Feature Correlation Heatmap")
plt.show()

# 3. Boxplots
num_cols = df.select_dtypes(include="number").columns.drop("stress_level")
for col in num_cols:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="stress_level", y=col, data=df, palette="Set3")
    plt.title(f"{col} Distribution by Stress Level")
    plt.show()


# ===============================
# Preprocessing
# ===============================

X = df.drop("stress_level", axis=1)
y = df["stress_level"]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y_cat = to_categorical(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM & BiLSTM
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_cat, test_size=0.2, random_state=42
)


# ===================================================
# LSTM MODEL
# ===================================================

print("\n================ LSTM MODEL TRAINING ================\n")

lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=False, input_shape=(1, X.shape[1])))
lstm_model.add(Dropout(0.3))
lstm_model.add(Dense(32, activation="relu"))
lstm_model.add(Dense(y_cat.shape[1], activation="softmax"))

lstm_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

lstm_model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train LSTM
history_lstm = lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Accuracy Plot
plt.figure(figsize=(8, 5))
plt.plot(history_lstm.history["accuracy"], label="Train Accuracy")
plt.plot(history_lstm.history["val_accuracy"], label="Validation Accuracy")
plt.title("LSTM Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Loss Plot
plt.figure(figsize=(8, 5))
plt.plot(history_lstm.history["loss"], label="Train Loss")
plt.plot(history_lstm.history["val_loss"], label="Validation Loss")
plt.title("LSTM Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Evaluation
y_pred = lstm_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nLSTM Classification Report:")
print(classification_report(y_true, y_pred_classes))


# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(6, 4))
disp.plot(cmap="Blues")
plt.title("LSTM Confusion Matrix")
plt.show()


# Save LSTM Model
lstm_model.save("lstm_stress_model.h5")
print("LSTM Model Saved Successfully!")


# ===================================================
# BiLSTM MODEL
# ===================================================

print("\n================ Bi-LSTM MODEL TRAINING ================\n")

bilstm_model = Sequential()
bilstm_model.add(
    Bidirectional(
        LSTM(64, return_sequences=False),
        input_shape=(1, X.shape[1])
    )
)
bilstm_model.add(Dropout(0.3))
bilstm_model.add(Dense(64, activation="relu"))
bilstm_model.add(Dropout(0.3))
bilstm_model.add(Dense(y_cat.shape[1], activation="softmax"))

bilstm_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

bilstm_model.summary()

# Train Bi-LSTM
history_bilstm = bilstm_model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Accuracy Plot
plt.figure(figsize=(8, 5))
plt.plot(history_bilstm.history["accuracy"], label="Train Accuracy")
plt.plot(history_bilstm.history["val_accuracy"], label="Validation Accuracy")
plt.title("Bi-LSTM Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Loss Plot
plt.figure(figsize=(8, 5))
plt.plot(history_bilstm.history["loss"], label="Train Loss")
plt.plot(history_bilstm.history["val_loss"], label="Validation Loss")
plt.title("Bi-LSTM Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Evaluation
y_pred_bi = bilstm_model.predict(X_test)
y_pred_bi_classes = np.argmax(y_pred_bi, axis=1)

print("\nBi-LSTM Classification Report:")
print(classification_report(y_true, y_pred_bi_classes))

# Confusion Matrix
cm_bi = confusion_matrix(y_true, y_pred_bi_classes)
disp_bi = ConfusionMatrixDisplay(confusion_matrix=cm_bi)
plt.figure(figsize=(6, 4))
disp_bi.plot(cmap="Greens")
plt.title("Bi-LSTM Confusion Matrix")
plt.show()


# Save Bi-LSTM Model
bilstm_model.save("bilstm_stress_model.h5")
print("Bi-LSTM Model Saved Successfully!")

# Save pre-processing tools
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Scaler & Label Encoder Saved Successfully!")
