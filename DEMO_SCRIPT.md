# VoiceAssist — 3–5 Minute Demonstration Script

**Setup before demo:** Open the deployed app URL, allow microphone permission when prompted. Have the text input box ready as backup.

---

**0:00 – 0:30 — Introduction**
"This is VoiceAssist, a voice-enabled student assistant. Unlike a simple keyword-matching bot, it uses a Bidirectional LSTM deep learning model that I trained myself to classify user intent. Let me walk through a few examples."

**0:30 – 1:15 — Demo 1: Greeting**
Say: **"Hello"**
Point out: recognized text appears → predicted intent `greeting` → high confidence → friendly response.
"Notice the pipeline: my voice was captured in the browser, sent as audio, transcribed to text, and that text was passed to the neural network which predicted the intent with a confidence score."

**1:15 – 2:00 — Demo 2: Academic Query**
Say: **"How can I prepare for my exams?"**
Point out: intent `study_help`, confidence percentage, and the generated study tip response.
"This shows the model generalizing — I never trained it on this *exact* sentence, only similar variations, so this demonstrates real learning rather than memorization."

**2:00 – 2:45 — Demo 3: Internship Query**
Say: **"How can I find an internship?"**
Point out: intent `internship`, response about the placement cell / Internshala.
Optionally click **"🔊 Speak Response"** here to demonstrate Text-to-Speech.

**2:45 – 3:30 — Demo 4: Unknown/Out-of-scope Question**
Say something unrelated, e.g.: **"What's the weather today?"**
Point out: confidence drops below the threshold shown in the sidebar → fallback response is triggered instead of a wrong guess.
"This is the confidence-threshold safeguard — the system admits uncertainty instead of confidently answering something it wasn't trained on."

**3:30 – 4:15 — Demo 5: Text Input (backup path)**
Type a question, e.g.: **"What is the library timing?"**, click Send.
"Both voice and text go through the exact same preprocessing, tokenization, and BiLSTM model — this is also the fallback path if microphone access isn't available during evaluation."

**4:15 – 5:00 — Wrap-up**
Briefly show the sidebar: model architecture, dataset info, supported intents, and the confidence threshold slider. Mention the evaluation results (accuracy, confusion matrix) from the report.
"The complete source code, dataset, trained model, evaluation plots, and report are all included in the submission."
