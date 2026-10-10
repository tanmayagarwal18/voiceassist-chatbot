# VoiceAssist – Voice-Enabled Deep Learning Chatbot

## Overview
VoiceAssist is a **Voice-Enabled Intelligent Student Assistant** built for a Speech and Language Processing laboratory assessment. It accepts speech (or typed text), classifies the user's intent using a trained **Bidirectional LSTM deep learning model**, and returns an appropriate response — optionally spoken aloud via Text-to-Speech.

This is a **prototype**: it uses a demo knowledge base of student-life intents (attendance, exams, placements, hostel, etc.) and is **not connected to any real college database**.

## Features
- 🎤 Voice input (browser microphone) and ⌨️ text input, both routed through the exact same NLP/DL pipeline
- 🧠 Custom-trained BiLSTM intent classifier (TensorFlow/Keras) — not a rule-based `if/else` bot, not an LLM API call
- 📊 Confidence score shown for every prediction, with a configurable confidence threshold and fallback response
- 🔊 Text-to-Speech playback of responses (gTTS)
- 💬 Session-based conversation history
- 🛡️ Graceful error handling for microphone, recognition, and model-loading failures

## System Architecture
```
User
 ↓
Microphone ──────────────┐
 ↓                        │
Speech Recognition        │
 ↓                        ↓
Recognized Text  <── Typed Text
 ↓
NLP Preprocessing (lowercase, punctuation removal, whitespace normalization)
 ↓
Tokenizer (vocabulary + sequence conversion) → Padding
 ↓
Embedding Layer
 ↓
Bidirectional LSTM
 ↓
Dropout → Dense (ReLU) → Dropout → Dense (Softmax)
 ↓
Intent + Confidence
 ↓
Confidence Threshold Check
 ↓
Response Generator (random choice from intent's response list, or fallback)
 ↓
Text Response ──→ Text-to-Speech ──→ Audio Output
```

## Technologies Used
| Component | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Voice capture | `streamlit-mic-recorder` (browser mic → WAV bytes) |
| Speech-to-Text | `SpeechRecognition` (Google Web Speech API) |
| Deep Learning | TensorFlow / Keras (Embedding → BiLSTM → Dense) |
| Evaluation | scikit-learn (accuracy, precision, recall, F1, confusion matrix) |
| Text-to-Speech | gTTS |

## Dataset
- File: `data/intents.json`
- **20 intents**, **264 training utterances** (greeting, goodbye, thanks, attendance, exams, assignments, courses, timetable, internship, placements, projects, library, hostel, fees, study_help, college_information, technical_support, academic_calendar, facilities, fallback)
- Each intent has ~10–18 realistic, conversationally varied patterns and 3–4 candidate responses (chosen randomly at inference time)

## Deep Learning Model
| Layer | Configuration |
|---|---|
| Embedding | vocab_size × 64, input length 20 |
| Bidirectional LSTM | 64 units |
| Dropout | 0.4 |
| Dense (ReLU) | 64 units |
| Dropout | 0.4 |
| Dense (Softmax) | 20 units (one per intent) |

- Optimizer: Adam · Loss: sparse categorical cross-entropy
- EarlyStopping on validation loss (patience = 10, restore best weights)
- Train/validation split: 80% / 20%, stratified by intent

## Speech Recognition
Because a deployed Streamlit app runs on a server with **no physical microphone**, audio is recorded **in the user's browser** via `streamlit-mic-recorder` and sent to the server as WAV bytes, which are then transcribed using `SpeechRecognition`'s Google Web Speech API integration. All failure modes (permission denied, unrecognized speech, empty audio, network failure) are handled and shown as friendly messages — the app never crashes.

## Text-to-Speech
`gTTS` converts the chatbot's text response into an MP3, played back with `st.audio()`. If TTS fails (e.g. no internet), the text response is still shown and the app keeps working.

## Installation
```bash
git clone <your-repo-url>
cd voiceassist
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training the Model
```bash
python train.py
```
This loads `data/intents.json`, preprocesses and tokenizes the text, trains the BiLSTM model, evaluates it, and saves:
- `models/chatbot_model.keras`, `models/tokenizer.pkl`, `models/label_encoder.pkl`, `models/responses.pkl`
- `results/accuracy.png`, `results/loss.png`, `results/confusion_matrix.png`, `results/metrics.txt`

## Running Locally
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`). Allow microphone access in the browser when prompted, or use the text input box.

## Project Structure
```
voiceassist/
├── app.py            # Streamlit UI
├── train.py          # Training pipeline
├── predict.py         # Inference pipeline (predict_intent)
├── preprocess.py       # Text cleaning utilities
├── chatbot.py          # Chatbot wrapper / sidebar helpers
├── speech.py            # Speech-to-text (browser mic → text)
├── tts.py                # Text-to-speech (gTTS)
├── data/intents.json      # Intent dataset
├── models/                 # Saved model, tokenizer, label encoder (after training)
├── results/                 # Saved plots and metrics (after training)
├── requirements.txt
├── README.md
└── .gitignore
```

## Results
*(Obtained by actually running `train.py` on this dataset — see `results/metrics.txt` for the raw output.)*

| Metric | Value |
|---|---|
| Dataset size | 264 utterances |
| Vocabulary size | 408 |
| Training samples | 211 |
| Validation samples | 53 |
| Accuracy | 62.3% |
| Precision (weighted) | 62.4% |
| Recall (weighted) | 62.3% |
| F1-score (weighted) | 58.3% |

See `results/accuracy.png`, `results/loss.png`, and `results/confusion_matrix.png` for training curves and the confusion matrix. Accuracy is modest because the validation split leaves only ~2–4 examples per class (see Limitations) — this is an honest result from the actual training run, not a target number.

## Deployment
Deployed on **Streamlit Community Cloud**. See `DEPLOYMENT.md` for full step-by-step instructions (GitHub setup, Streamlit Cloud configuration, testing microphone access, and obtaining the public URL).

**Live App URL:** `<insert your deployed URL here after deployment>`

## Future Enhancements
- Transformer-based intent classification (e.g. a small fine-tuned BERT) for higher accuracy
- Multilingual speech recognition
- Personalized student profiles and real database integration
- Retrieval-augmented generation for open-ended questions
- Larger, more diverse conversational dataset
- Emotion detection from voice
- Context-aware multi-turn conversations

## Author
`<Your Name>` — B.Tech Computer Science, Speech and Language Processing Laboratory Assessment.
