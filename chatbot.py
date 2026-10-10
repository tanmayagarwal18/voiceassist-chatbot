"""
chatbot.py
----------
Thin chatbot-logic layer used by the Streamlit UI.

Keeps app.py focused on presentation, and keeps the DL inference details
inside predict.py. This module also exposes the list of supported intents
(read from data/intents.json) for display in the UI sidebar.
"""

import json
import os

from predict import IntentPredictor, DEFAULT_CONFIDENCE_THRESHOLD

INTENTS_PATH = os.path.join("data", "intents.json")


def get_supported_intents():
    """Return the list of intent tags defined in the dataset (for the sidebar)."""
    try:
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [i["tag"] for i in data["intents"]]
    except Exception:
        return []


def get_dataset_stats():
    """Small helper for the sidebar 'Dataset Information' panel."""
    try:
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        num_intents = len(data["intents"])
        num_utterances = sum(len(i["patterns"]) for i in data["intents"])
        return {"num_intents": num_intents, "num_utterances": num_utterances}
    except Exception:
        return {"num_intents": 0, "num_utterances": 0}


class Chatbot:
    """
    High-level chatbot object used by the Streamlit app.
    Wraps IntentPredictor and adds simple guard-rails around loading errors.
    """

    def __init__(self):
        self.ready = False
        self.error_message = None
        try:
            self.predictor = IntentPredictor()
            self.ready = True
        except FileNotFoundError as e:
            self.predictor = None
            self.error_message = str(e)

    def get_response(self, user_text: str, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
        """
        Given raw user text (from voice or typing), run the full DL pipeline
        and return a dict with intent, confidence, response, is_fallback.
        Both input methods (voice-transcribed text and typed text) go
        through this exact same function -> same NLP/DL pipeline.
        """
        if not self.ready:
            return {
                "intent": "system_error",
                "confidence": 0.0,
                "response": "The chatbot model isn't available right now. "
                            "Please make sure the model has been trained "
                            "(run `python train.py`) and try again.",
                "is_fallback": True,
            }
        return self.predictor.predict_intent(user_text, threshold=threshold)
