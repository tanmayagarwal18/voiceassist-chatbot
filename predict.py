"""
predict.py
----------
Inference pipeline: loads the trained model/tokenizer/label encoder once,
and exposes predict_intent(text) for use by chatbot.py and app.py.
"""

import os
import pickle
import random

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import clean_text, MAX_SEQUENCE_LENGTH

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
RESPONSES_PATH = os.path.join(MODEL_DIR, "responses.pkl")

# Default confidence threshold - configurable from the Streamlit sidebar.
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

FALLBACK_RESPONSES = [
    "I'm not completely sure I understood that. Could you rephrase your question?",
    "I'm designed to help with student-related queries. Could you ask something about college, courses, or exams?",
]


class IntentPredictor:
    """
    Loads model artifacts once and provides predict_intent(text, threshold).
    Raises a clear, catchable error if model files are missing so the UI
    can show a friendly message instead of crashing.
    """

    def __init__(self):
        missing = [p for p in [MODEL_PATH, TOKENIZER_PATH, ENCODER_PATH, RESPONSES_PATH]
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing trained model artifact(s): " + ", ".join(missing) +
                ". Please run `python train.py` first."
            )

        self.model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)
        with open(RESPONSES_PATH, "rb") as f:
            self.responses_map = pickle.load(f)

    def predict_intent(self, text: str, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
        """
        Full inference pipeline for a single utterance.

        Steps: preprocess -> tokenize -> pad -> model forward pass ->
               softmax probabilities -> argmax -> confidence threshold ->
               response selection.

        Returns:
            {
                "intent": str,
                "confidence": float,     # 0.0 - 1.0
                "response": str,
                "is_fallback": bool
            }
        """
        if text is None or not text.strip():
            return {
                "intent": "empty_input",
                "confidence": 0.0,
                "response": "I didn't catch anything. Could you say or type your question again?",
                "is_fallback": True,
            }

        cleaned = clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

        probabilities = self.model.predict(padded, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        predicted_tag = self.label_encoder.inverse_transform([best_index])[0]

        if confidence >= threshold and predicted_tag in self.responses_map:
            response = random.choice(self.responses_map[predicted_tag])
            return {
                "intent": predicted_tag,
                "confidence": confidence,
                "response": response,
                "is_fallback": False,
            }
        else:
            return {
                "intent": predicted_tag,  # still shown for transparency/debugging
                "confidence": confidence,
                "response": random.choice(FALLBACK_RESPONSES),
                "is_fallback": True,
            }
