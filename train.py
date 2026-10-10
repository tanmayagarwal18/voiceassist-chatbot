"""
train.py
--------
Trains a Bidirectional LSTM intent-classification model on data/intents.json.

Pipeline:
    intents.json -> patterns/labels -> clean_text -> Tokenizer -> pad_sequences
    -> LabelEncoder -> train/val split (stratified) -> BiLSTM model
    -> train with EarlyStopping -> evaluate (accuracy/precision/recall/F1/confusion matrix)
    -> save model, tokenizer, label encoder, plots, metrics report

Run:
    python train.py
"""

import json
import os
import pickle
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for saving plots without a display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import clean_text, MAX_SEQUENCE_LENGTH, OOV_TOKEN

# ----------------------------- Reproducibility -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------- Paths ---------------------------------------
DATA_PATH = os.path.join("data", "intents.json")
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------- Hyperparameters ------------------------------
EMBEDDING_DIM = 64
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.4
EPOCHS = 80
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2  # 80/20 train/validation split


def load_dataset(path):
    """Load intents.json and flatten into (texts, labels, responses_map)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    responses_map = {}

    for intent in data["intents"]:
        tag = intent["tag"]
        responses_map[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            labels.append(tag)

    return texts, labels, responses_map


def build_model(vocab_size, num_classes):
    """Embedding -> BiLSTM -> Dropout -> Dense(ReLU) -> Dropout -> Dense(Softmax)."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                  input_length=MAX_SEQUENCE_LENGTH, name="embedding"),
        Bidirectional(LSTM(LSTM_UNITS), name="bilstm"),
        Dropout(DROPOUT_RATE, name="dropout_1"),
        Dense(DENSE_UNITS, activation="relu", name="dense_relu"),
        Dropout(DROPOUT_RATE, name="dropout_2"),
        Dense(num_classes, activation="softmax", name="output_softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    """Save training vs validation accuracy & loss curves."""
    # Accuracy
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy.png"), dpi=150)
    plt.close()

    # Loss
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 9))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                      ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black",
                      fontsize=7)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def main():
    print("Loading dataset...")
    texts, labels, responses_map = load_dataset(DATA_PATH)

    num_intents = len(set(labels))
    print(f"Dataset size: {len(texts)}")
    print(f"Number of intents: {num_intents}")

    # ---------------- Label encoding ----------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # ---------------- Tokenization & padding ----------------
    tokenizer = Tokenizer(oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of classes: {num_intents}")

    # ---------------- Train / validation split (stratified) ----------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=SEED, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # ---------------- Build & train model ----------------
    model = build_model(vocab_size, num_intents)
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    # ---------------- Evaluation ----------------
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="weighted", zero_division=0
    )

    print("\n===== EVALUATION RESULTS (VALIDATION SET) =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    class_names = list(label_encoder.classes_)
    report = classification_report(
        y_val, y_pred, target_names=class_names, zero_division=0
    )
    print("\nClassification report:\n", report)

    cm = confusion_matrix(y_val, y_pred)

    # ---------------- Save plots ----------------
    plot_history(history)
    plot_confusion_matrix(cm, class_names)

    # ---------------- Save metrics to a text file (for the report) ----------------
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write("VoiceAssist - Model Evaluation Metrics\n")
        f.write("=======================================\n")
        f.write(f"Dataset size: {len(texts)}\n")
        f.write(f"Number of intents: {num_intents}\n")
        f.write(f"Vocabulary size: {vocab_size}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)

    # ---------------- Save model, tokenizer, label encoder ----------------
    model.save(os.path.join(MODEL_DIR, "chatbot_model.keras"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(MODEL_DIR, "responses.pkl"), "wb") as f:
        pickle.dump(responses_map, f)

    print("\nSaved model, tokenizer, label encoder, responses, plots and metrics.")
    print("Files written to:", MODEL_DIR, "and", RESULTS_DIR)


if __name__ == "__main__":
    main()
