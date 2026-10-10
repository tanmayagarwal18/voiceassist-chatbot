
"""
app.py
------
VoiceAssist - Voice-Enabled Intelligent Student Assistant

Run locally with:
    streamlit run app.py
"""

import streamlit as st

from chatbot import Chatbot, get_supported_intents, get_dataset_stats
from speech import transcribe_audio_bytes
from tts import synthesize_speech
from predict import DEFAULT_CONFIDENCE_THRESHOLD

# Microphone component
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

st.set_page_config(
    page_title="VoiceAssist",
    page_icon="🎙️",
    layout="centered"
)

# ----------------------------- Session state --------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = Chatbot()

chatbot = st.session_state.chatbot

# ----------------------------- Sidebar --------------------------------------
with st.sidebar:
    st.header(" About Project")
    st.write(
        "**VoiceAssist** is a prototype voice-enabled student assistant "
        "built for a Speech & Language Processing lab assessment. It uses "
        "a trained Deep Learning model (BiLSTM) for intent classification. "
        "It is not connected to any real college database."
    )

    st.header(" Model Information")
    st.write(
        "- Architecture: Embedding → BiLSTM → Dropout → Dense(ReLU) "
        "→ Dropout → Softmax\n"
        "- Framework: TensorFlow / Keras\n"
        "- Task: Intent classification (multi-class)"
    )

    if not chatbot.ready:
        st.error(chatbot.error_message)

    st.header(" Dataset Information")
    stats = get_dataset_stats()
    st.write(f"- Intents: **{stats['num_intents']}**")
    st.write(f"- Training utterances: **{stats['num_utterances']}**")
    st.caption("Demo dataset only - see report for limitations.")

    st.header(" Supported Intents")
    intents_list = get_supported_intents()
    st.write(", ".join(intents_list) if intents_list else "Not available")

    st.header(" Confidence Threshold")
    threshold = st.slider(
        "Minimum confidence to trust the model's prediction",
        min_value=0.10,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
    )
    st.caption(
        "Below this threshold, the assistant gives a fallback "
        "response instead of guessing."
    )

    if st.button(" Clear Conversation History"):
        st.session_state.history = []
        st.rerun()

# ----------------------------- Header ---------------------------------------
st.title("🎙️ VoiceAssist")
st.subheader("Voice-Enabled Student Assistant")
st.write("Ask your question using your voice or type it below.")
st.divider()


def handle_query(user_text: str, source_label: str):
    """Run the shared NLP/DL pipeline and append the turn to history."""
    result = chatbot.get_response(user_text, threshold=threshold)

    st.session_state.history.append({
        "user_input_source": source_label,
        "recognized_text": user_text,
        "intent": result["intent"],
        "confidence": result["confidence"],
        "response": result["response"],
        "is_fallback": result["is_fallback"],
    })


# ----------------------------- Voice input ----------------------------------
st.markdown("### 🎤 Voice Input")

if MIC_AVAILABLE:
    st.caption(
        "Click Start Speaking, say your question, then click Stop Recording."
    )

    audio = mic_recorder(
        start_prompt="🎤 Start Speaking",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        use_container_width=True,
        format="wav",
        key="recorder",
    )

    if audio and audio.get("bytes"):
        try:
            with st.spinner("Transcribing speech..."):
                stt_result = transcribe_audio_bytes(audio["bytes"])

            if stt_result.get("success") and stt_result.get("text", "").strip():
                recognized_text = stt_result["text"].strip()

                st.success(f'Recognized Speech: "{recognized_text}"')

                handle_query(recognized_text, "voice")

                # Refresh to display the latest chatbot response
                st.rerun()

            else:
                st.warning(
                    stt_result.get(
                        "error",
                        "Speech could not be recognized. Please try again."
                    )
                )

        except Exception as e:
            st.error(f"Could not process audio: {e}")

else:
    st.warning(
        "The microphone component is unavailable. "
        "Install it using %pip install streamlit-mic-recorder "
        "in your Jupyter Notebook, then restart Streamlit."
    )

# ----------------------------- Text input -----------------------------------
st.markdown("###  Type Your Question")

with st.form("text_form", clear_on_submit=True):
    typed_text = st.text_input("Type your question here:", "")
    submitted = st.form_submit_button("Send")

    if submitted:
        if typed_text.strip():
            handle_query(typed_text.strip(), "text")
            st.rerun()
        else:
            st.warning("Please type something before sending.")

st.divider()

# ----------------------------- Latest result panel --------------------------
if st.session_state.history:
    latest = st.session_state.history[-1]

    st.markdown("###  AI Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Intent", latest["intent"])

    with col2:
        st.metric(
            "Confidence",
            f"{latest['confidence'] * 100:.1f}%"
        )

    if latest["is_fallback"]:
        st.caption(
            "⚠️ Confidence below threshold - fallback response used."
        )

    st.markdown("### 🤖 Assistant")
    st.info(latest["response"])

    if st.button("🔊 Speak Response"):
        try:
            with st.spinner("Generating speech..."):
                audio_bytes, tts_error = synthesize_speech(
                    latest["response"]
                )

            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            else:
                st.warning(tts_error)

        except Exception as e:
            st.warning(f"Text-to-speech failed: {e}")

st.divider()

# ----------------------------- Conversation history -------------------------
st.markdown("###  Conversation History")

if not st.session_state.history:
    st.caption(
        "No conversation yet - try the voice or text input above."
    )
else:
    for turn in reversed(st.session_state.history):
        st.markdown(
            f"**User** ({turn['user_input_source']}): "
            f"{turn['recognized_text']}"
        )

        st.markdown(
            f"**Assistant** (intent: `{turn['intent']}`, "
            f"confidence: `{turn['confidence'] * 100:.1f}%`): "
            f"{turn['response']}"
        )

        st.markdown("---")
