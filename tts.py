"""
tts.py
------
Text-to-Speech module.

Uses gTTS (Google Text-to-Speech) which is lightweight, requires no local
model weights, and produces an MP3 in-memory that Streamlit can play
directly with st.audio(). This is a server-side approach (no browser
JavaScript speech synthesis needed), which keeps the deployment simple.

If TTS fails for any reason (no internet, service issue, etc.), the app
must keep working with just the text response - TTS is optional.
"""

import io

from gtts import gTTS


def synthesize_speech(text: str):
    """
    Convert text to speech audio bytes (MP3).

    Returns:
        (audio_bytes, error) where exactly one of them is None/falsy.
        - audio_bytes: bytes object suitable for st.audio(audio_bytes, format="audio/mp3")
        - error: human-readable error string if synthesis failed
    """
    if not text or not text.strip():
        return None, "There is no response text to speak."

    try:
        tts = gTTS(text=text, lang="en")
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read(), None
    except Exception as e:
        return None, f"Text-to-speech is temporarily unavailable ({e}). The text response above is still valid."
