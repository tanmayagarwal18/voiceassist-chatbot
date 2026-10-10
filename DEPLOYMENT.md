# Deployment Instructions — Streamlit Community Cloud

## 1. Create a GitHub repository
1. Go to https://github.com/new and create a new repository (e.g. `voiceassist`). Public or private both work with Streamlit Cloud (private needs your GitHub linked).
2. On your machine, inside the `voiceassist/` project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: VoiceAssist project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/voiceassist.git
   git push -u origin main
   ```

## 2. Train the model before pushing (important)
Run `python train.py` locally first so that `models/chatbot_model.keras`, `models/tokenizer.pkl`, `models/label_encoder.pkl`, and `models/responses.pkl` exist and get committed to the repo — Streamlit Cloud does not train the model for you; it only runs `app.py`.

## 3. Verify requirements.txt
Make sure `requirements.txt` is present at the repo root (already included in this project) and lists only the packages actually used.

## 4. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository, branch (`main`), and main file path (`app.py`).
4. Click **"Deploy"**. The first build can take a few minutes (installing TensorFlow).

## 5. Configuring secrets (if needed)
This project doesn't require any API keys (Google Web Speech API used by `SpeechRecognition` and gTTS both work without a key for light usage). If you later add a paid/keyed service, add it under your app's **Settings → Secrets** in the Streamlit Cloud dashboard as:
```toml
MY_API_KEY = "..."
```
and read it in code with `st.secrets["MY_API_KEY"]`.

## 6. Testing microphone access
- Microphone recording happens **in the browser**, not on the server, via the `streamlit-mic-recorder` component.
- The browser will prompt for microphone permission the first time you click "Start Speaking" — click **Allow**.
- Streamlit Cloud apps are served over HTTPS, which is required by browsers for microphone access to work (it will **not** work over plain HTTP, only `localhost` or HTTPS).
- If permission is denied or no microphone is available, the app falls back cleanly to the **text input box** — the whole demo can be given without voice if necessary.

## 7. Obtaining the public URL
After deployment finishes, Streamlit Cloud shows a URL like:
```
https://<your-app-name>-<random-id>.streamlit.app
```
This is your public, shareable link — use it as your "Live application link" deliverable.

## Common deployment issues & fixes
| Issue | Fix |
|---|---|
| Build fails on TensorFlow install | Make sure `tensorflow-cpu` (not full `tensorflow`) is in requirements.txt — smaller and faster to install |
| App boots but shows "Missing trained model artifact(s)" | You forgot to run `train.py` and commit the `models/` folder before pushing |
| Mic button does nothing | Check the browser address bar for a blocked microphone icon; must be HTTPS or localhost |
| App is slow on first load | Normal — TensorFlow model loading takes a few seconds on Streamlit Cloud's free tier |
