"""
preprocess.py
--------------
Text preprocessing utilities shared by training and inference.

Pipeline implemented here:
    raw text -> lowercase -> punctuation removal -> whitespace normalization

We deliberately keep preprocessing LIGHT:
  - We do NOT remove stopwords (words like "how", "what", "not" carry
    important intent signal, e.g. "how much attendance" vs "attendance").
  - We do NOT stem/lemmatize aggressively, since the dataset is small and
    a BiLSTM over raw (but cleaned) tokens works well for short queries.
Tokenization, vocabulary building, sequence conversion and padding are
handled by Keras' Tokenizer + pad_sequences (fit during training and
reused, unchanged, during inference).
"""

import re
import string

# Maximum sequence length (in tokens) used for padding.
# Chosen based on the fact that student queries are short (typically < 12 words).
MAX_SEQUENCE_LENGTH = 20

# Special token used for out-of-vocabulary words at inference time.
OOV_TOKEN = "<OOV>"


def clean_text(text: str) -> str:
    """
    Clean a single raw utterance before tokenization.

    Steps:
      1. Lowercase conversion  -> normalizes case differences ("Hello" == "hello")
      2. Punctuation removal   -> "How's it going?" -> "hows it going"
      3. Whitespace collapsing -> removes double spaces / tabs / newlines
      4. Strip leading/trailing whitespace

    We keep numbers and apostrophised words as-is (after punctuation strip)
    since removing too much can erase meaning for short queries.
    """
    if text is None:
        return ""

    text = text.lower()
    # Remove punctuation (keep letters, digits, and whitespace)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_corpus(texts):
    """Apply clean_text to a list of utterances."""
    return [clean_text(t) for t in texts]
