# VoiceAssist — Viva Questions & Answers

1. **What is Speech Recognition?**
   The process of converting spoken audio into written text, so a computer can process what was said.

2. **Why did you use Speech Recognition?**
   The assessment requires a voice-enabled chatbot, so speech must first be converted to text before any NLP or deep learning model can process it.

3. **What is NLP?**
   Natural Language Processing — techniques that let computers understand, interpret, and generate human language.

4. **What is intent classification?**
   A supervised learning task where a piece of text is assigned to one of a fixed set of categories ("intents") representing what the user wants, e.g. "attendance", "exams", "greeting".

5. **Why did you use Deep Learning instead of simple keyword matching?**
   Keyword matching (`if "hello" in text`) is brittle and fails on paraphrased or reordered queries. A trained neural network learns patterns in word usage and generalizes to unseen phrasings of the same intent.

6. **Why BiLSTM specifically?**
   An LSTM can model sequential dependencies in text; a *Bidirectional* LSTM reads the sentence both forward and backward, so it captures context from words that come both before and after a given word, which improves understanding of short queries.

7. **What is an LSTM?**
   Long Short-Term Memory — a type of recurrent neural network designed to remember information over longer sequences by using gates (input, forget, output) that control what information is kept or discarded.

8. **Why is Bidirectional LSTM better than a plain LSTM here?**
   Because student queries are short, understanding words at the end of the sentence (e.g. "...attendance?") together with words at the start improves classification accuracy compared to reading in only one direction.

9. **What is an embedding layer?**
   A layer that maps each word (represented as an integer index) to a dense, trainable vector of fixed size, so that words with similar usage end up with similar vector representations.

10. **Why is tokenization required?**
    Neural networks operate on numbers, not raw text, so tokenization converts words into integer indices (based on a vocabulary built from the training data) before they can be fed into the model.

11. **What is padding?**
    Since sentences have different lengths but neural networks need fixed-size input, padding adds zeros (or truncates) so every input sequence has the same length (20 tokens in this project).

12. **What is softmax?**
    An activation function used in the output layer that converts raw model scores ("logits") into a probability distribution over all classes, where all probabilities sum to 1.

13. **What is the purpose of dropout?**
    A regularization technique that randomly "drops" (zeroes out) a fraction of neurons during training, which prevents the model from over-relying on specific neurons and reduces overfitting.

14. **What loss function did you use, and why?**
    Sparse categorical cross-entropy, because it's designed for multi-class classification with integer-encoded labels (rather than one-hot vectors).

15. **Why did you use the Adam optimizer?**
    Adam adapts the learning rate for each parameter individually and generally converges faster and more reliably than plain SGD, especially for small-to-medium sized datasets like this one.

16. **What is overfitting?**
    When a model learns the training data too specifically (including its noise) and performs well on training data but poorly on unseen validation/test data.

17. **How did you try to prevent overfitting?**
    By using Dropout layers, EarlyStopping (which halts training once validation loss stops improving), and keeping the model architecture compact relative to the dataset size.

18. **What is a confusion matrix?**
    A table showing the counts of correct and incorrect predictions for each class, with true labels on one axis and predicted labels on the other — it reveals which intents are being confused with each other.

19. **What are precision, recall, and F1-score?**
    Precision is the proportion of predicted positives that are actually correct; recall is the proportion of actual positives that were correctly identified; F1-score is the harmonic mean of precision and recall, balancing both.

20. **How does the chatbot generate responses?**
    Once an intent is predicted with sufficient confidence, a response is randomly selected from a pre-written list of responses associated with that intent tag.

21. **What happens when the model is uncertain?**
    If the model's confidence for its top prediction is below the configured threshold, the system returns a fallback response ("I'm not completely sure...") instead of guessing.

22. **Why did you implement a confidence threshold?**
    To avoid confidently giving a wrong or irrelevant answer to a completely unrelated or out-of-scope question — it makes the system safer and more trustworthy.

23. **What happens if speech recognition fails?**
    The app catches the specific error (no speech detected, unrecognizable audio, network failure) and shows a clear message, while the text input box remains available as a backup — the app never crashes.

24. **Why did you use Streamlit?**
    Streamlit allows rapid development of a clean, interactive Python-based web UI with built-in components (like audio widgets), simplifying deployment for a laboratory project without building a separate frontend.

25. **How did you deploy the project?**
    The code was pushed to a GitHub repository and deployed on Streamlit Community Cloud, which automatically builds and hosts the app from `requirements.txt` and `app.py`, providing a public HTTPS URL.

26. **What are the limitations of your project?**
    A small dataset (264 utterances across 20 intents), no real database integration, single-turn (not multi-turn) conversation, and dependency on an external speech recognition API and internet connectivity.

27. **How could the project be improved using Transformers?**
    A pretrained Transformer (e.g. BERT/DistilBERT) fine-tuned on the same intents could likely achieve higher accuracy with the same or even less training data, since it starts from rich, pretrained language representations rather than learning embeddings from scratch.

28. **What is the difference between speech recognition and NLP?**
    Speech recognition converts audio into text (an acoustic/signal-processing problem); NLP then processes that text to extract meaning, intent, or generate a response (a language-understanding problem). They are sequential, complementary stages.

29. **What is the difference between intent classification and response generation?**
    Intent classification decides *what category* the user's message belongs to (a classification problem with a fixed label set); response generation decides *what to say back*, which in this project is a simple lookup/random-choice from pre-written responses tied to the predicted intent.

30. **Why didn't you use an LLM (like GPT) for this project?**
    The assessment specifically requires demonstrating understanding of a deep learning model built and trained by the student, not simply calling an external API. A custom BiLSTM intent classifier shows genuine understanding of NLP preprocessing, embeddings, sequence modelling, and evaluation — which a wrapped LLM API call would not demonstrate.
