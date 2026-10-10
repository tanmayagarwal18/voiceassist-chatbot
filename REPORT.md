# VoiceAssist: A Voice-Enabled Chatbot Using Speech Recognition and Deep Learning

**Speech and Language Processing Laboratory Assessment**

---

## 2. Abstract
This project presents VoiceAssist, a voice-enabled intelligent student assistant that combines speech recognition, natural language processing, and deep learning to answer common student queries. The system captures voice input through a browser microphone, transcribes it to text using the Google Web Speech API, and classifies the underlying intent using a Bidirectional LSTM (BiLSTM) neural network trained on a custom dataset of 264 labelled student-life utterances spanning 20 intent categories. Based on the predicted intent and a configurable confidence threshold, the system generates a suitable response, which can optionally be converted back to speech via Text-to-Speech. The complete application is built with Streamlit and TensorFlow/Keras and deployed publicly on Streamlit Community Cloud. On a held-out validation split, the trained model achieved an accuracy of 62.3% and a weighted F1-score of 58.3%, reflecting a working proof-of-concept intent classifier trained on a deliberately small, hand-crafted academic dataset.

## 3. Introduction
Conversational AI systems allow users to interact with software using natural language instead of rigid menus or commands. Two building blocks make this possible: **Speech Recognition**, which converts spoken audio into text, and **Natural Language Processing (NLP)**, which extracts meaning from that text. A common and practical NLP task for task-oriented assistants is **intent classification** — determining *what the user wants* from a fixed set of possible intents, rather than generating open-ended text. **Deep Learning**, particularly recurrent architectures like LSTMs, is well suited to this task because it can learn distributed representations of words and their sequential context, generalizing beyond exact keyword matching. This project combines these components into a voice-enabled interface, demonstrating an end-to-end pipeline from raw audio to a spoken response.

## 4. Problem Statement
Develop, implement, and deploy an online voice-enabled chatbot using speech recognition and deep learning. The system must accept voice input, convert it to text, classify the intent using a trained deep learning model, generate an appropriate response, display the recognized text and response, optionally speak the response aloud, and be hosted online with public access — supported by source code and a technical report.

## 5. Objectives
1. Design a structured, diverse intent-classification dataset covering common student queries.
2. Implement a complete NLP preprocessing pipeline (cleaning, tokenization, padding, label encoding).
3. Build and train a BiLSTM-based deep learning model for intent classification.
4. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
5. Integrate speech-to-text and text-to-speech into a single conversational pipeline.
6. Build an accessible, visually clear Streamlit interface supporting both voice and text input.
7. Deploy the system publicly and document it for reproducibility and viva evaluation.

## 6. Dataset
The dataset (`data/intents.json`) is a hand-crafted, JSON-structured intent dataset containing **20 intents** and **264 total utterances** (an average of ~13 utterances per intent, ranging from 10 to 18). Each intent entry has a `tag`, a list of `patterns` (example user utterances), and a list of candidate `responses`. Intents cover: greeting, goodbye, thanks, attendance, exams, assignments, courses, timetable, internship, placements, projects, library, hostel, fees, study_help, college_information, technical_support, academic_calendar, facilities, and fallback (for out-of-scope queries). Patterns deliberately include realistic, conversational phrasing variety (e.g. "How much attendance do I need?", "What happens if my attendance is low?", "Attendance shortage what to do") rather than only grammatically perfect sentences, so the model must learn semantic patterns rather than memorize fixed strings. The dataset was split 80% (211 samples) / 20% (53 samples) for training and validation, using stratified sampling to preserve class proportions.

## 7. Methodology
The end-to-end pipeline is:
```
Speech Input → Speech Recognition (Google Web Speech API) → Text Preprocessing
→ Tokenization → Padding → Embedding → Bidirectional LSTM → Dropout → Dense →
Softmax → Intent Classification → Confidence Threshold → Response Generation → TTS
```
Text preprocessing lowercases input, strips punctuation, and normalizes whitespace, while deliberately avoiding stopword removal or stemming, since short queries rely on function words (e.g. "how", "what") for intent signal. A Keras `Tokenizer` builds the vocabulary and converts text to integer sequences, which are padded/truncated to a fixed length of 20 tokens. Labels are integer-encoded with scikit-learn's `LabelEncoder`.

## 8. Model Architecture
| Layer | Configuration | Purpose |
|---|---|---|
| Embedding | vocab_size (408) × 64 dims, input length 20 | Learns a dense vector representation for each word |
| Bidirectional LSTM | 64 units (forward + backward) | Captures sequential context from both directions of the sentence |
| Dropout | rate 0.4 | Regularization to reduce overfitting on the small dataset |
| Dense (ReLU) | 64 units | Learns non-linear combinations of the LSTM's output features |
| Dropout | rate 0.4 | Further regularization before the final classification layer |
| Dense (Softmax) | 20 units (one per intent) | Outputs a probability distribution over all intents |

The model was compiled with the **Adam** optimizer and **sparse categorical cross-entropy** loss (appropriate for integer-encoded multi-class labels), trained for up to 80 epochs with a batch size of 16, and used **EarlyStopping** (monitoring validation loss, patience 10, restoring best weights) to avoid unnecessary training once validation performance stopped improving.

## 9. Implementation
The project is organized into clearly separated modules:
- **`preprocess.py`** — text cleaning shared by training and inference.
- **`train.py`** — loads the dataset, tokenizes/pads/encodes it, builds and trains the BiLSTM model, evaluates it, and saves the model, tokenizer, label encoder, plots, and metrics.
- **`predict.py`** — loads the saved artifacts once and exposes `predict_intent(text, threshold)`, implementing the confidence-threshold fallback logic.
- **`chatbot.py`** — a thin wrapper exposing dataset stats and a `Chatbot` class used by the UI, so both voice and typed input reuse exactly the same DL pipeline.
- **`speech.py`** — transcribes browser-recorded WAV audio to text via `SpeechRecognition`, handling empty audio, unrecognizable speech, and network failures.
- **`tts.py`** — converts response text to speech using gTTS, failing gracefully if unavailable.
- **`app.py`** — the Streamlit UI: sidebar (about, model info, supported intents, confidence slider, dataset info), voice/text input, the AI analysis panel (intent + confidence), the assistant's response with a "Speak Response" button, and conversation history.

## 10. Results
Obtained by running `train.py` on the actual dataset (see `results/metrics.txt` for raw output):

| Metric | Value |
|---|---|
| Accuracy | 0.6226 (62.3%) |
| Precision (weighted) | 0.6236 |
| Recall (weighted) | 0.6226 |
| F1-score (weighted) | 0.5829 |
| Validation samples | 53 (out of 264 total) |

Training/validation accuracy and loss curves are saved at `results/accuracy.png` and `results/loss.png`; the confusion matrix is saved at `results/confusion_matrix.png`. The per-class classification report (in `results/metrics.txt`) shows strong performance on intents with more distinctive vocabulary (e.g. `attendance`, `greeting`, `goodbye`) and weaker performance on intents with only 2 validation examples each and overlapping vocabulary (e.g. `timetable`, `academic_calendar`, `technical_support`, `fallback`).

## 11. Deployment
The application is deployed on **Streamlit Community Cloud**, which builds directly from the project's GitHub repository and serves the Streamlit app over HTTPS (required for browser microphone access). Full step-by-step deployment instructions are provided in `DEPLOYMENT.md`.

**Live application URL:** `<insert your deployed Streamlit Cloud URL here>`

## 12. Limitations
- The dataset is small (264 utterances across 20 classes), leaving only 2–4 validation examples per class, which makes the reported metrics somewhat noisy/unstable.
- Intent classification with a fixed label set cannot handle genuinely novel or compound questions well.
- Speech recognition accuracy depends on internet connectivity, accent, and background noise, since it relies on an external Web Speech API.
- The knowledge base is a static, hand-written demo — not connected to any real college database, timetable system, or student records.
- The system cannot maintain multi-turn context (e.g. follow-up questions referring to a previous answer).

## 13. Future Scope
- Replace or augment the BiLSTM with a small fine-tuned Transformer (e.g. DistilBERT) for improved accuracy with more data.
- Support multilingual speech recognition for regional languages.
- Integrate personalized student profiles and a real academic database.
- Explore retrieval-augmented generation (RAG) for open-ended, out-of-scope questions.
- Expand the dataset substantially with more utterances per intent, collected from real student queries.
- Add emotion/sentiment detection from voice tone.
- Support context-aware, multi-turn conversations rather than single-turn intent classification.

## 14. Conclusion
VoiceAssist demonstrates a complete, working voice-enabled chatbot pipeline — from browser-captured speech, through a genuinely trained deep learning intent classifier, to a spoken response — built with an intermediate-level, deployable technology stack (Streamlit, TensorFlow/Keras, SpeechRecognition, gTTS). While the reported accuracy (62.3%) reflects the constraints of a small academic dataset, the project fulfills the assessment's core requirement: to design and deploy a system that genuinely applies speech processing, NLP, and deep learning together, rather than relying on simple keyword matching or an external LLM API.
