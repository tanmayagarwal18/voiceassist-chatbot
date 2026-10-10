"""
speech.py
---------
Speech-to-text module.

IMPORTANT DEPLOYMENT NOTE:
When a Streamlit app is deployed to the cloud, the Python process runs on a
remote server that has NO microphone of its own. Voice must therefore be
captured in the USER's BROWSER and sent to the server as audio bytes. We use
the `streamlit-mic-recorder` component for this (records in-browser, returns
WAV bytes to Python), and then run Google's free Web Speech API through the
`SpeechRecognition` library to transcribe those bytes to text.

Pipeline demonstrated:
    Browser Microphone -> WAV bytes -> SpeechRecognition (Google Web Speech API)
    -> recognized text

All failure modes are handled explicitly and return a structured result
instead of raising, so the Streamlit UI never crashes on a speech error.
"""

import io

import speech_recognition as sr

recognizer = sr.Recognizer()


def transcribe_audio_bytes(audio_bytes: bytes) -> dict:
    """
    Convert recorded audio (WAV bytes from the browser mic widget) into text.

    Returns:
        {
            "success": bool,
            "text": str,         # recognized text (empty if failed)
            "error": str | None  # human-readable error message if failed
        }
    """
    if not audio_bytes:
        return {
            "success": False,
            "text": "",
            "error": "No audio was captured. Please check your microphone permissions and try again.",
        }

    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            # Briefly calibrate for ambient/background noise before recording ends
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)
    except Exception:
        return {
            "success": False,
            "text": "",
            "error": "Could not read the recorded audio. Please try recording again.",
        }

    try:
        text = recognizer.recognize_google(audio_data)
        if not text.strip():
            return {
                "success": False,
                "text": "",
                "error": "No speech was detected. Please speak clearly and try again.",
            }
        return {"success": True, "text": text, "error": None}

    except sr.UnknownValueError:
        # Speech was present but not understandable (noise, mumbling, silence)
        return {
            "success": False,
            "text": "",
            "error": "Sorry, I couldn't understand the audio. Please try speaking more clearly, "
                     "or use the text input box instead.",
        }
    except sr.RequestError:
        # Network/service failure while contacting the speech recognition API
        return {
            "success": False,
            "text": "",
            "error": "The speech recognition service is currently unreachable "
                     "(check your internet connection). You can use the text input box instead.",
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "error": f"An unexpected error occurred during speech recognition: {e}",
        }
