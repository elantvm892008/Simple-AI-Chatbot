# Simple AI Chatbot

A beginner-friendly FAQ chatbot using TF-IDF and cosine similarity to match a user's question with a small set of stored questions.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – Chat interface and question matching
- `requirements.txt` – Dependencies

## How it works
1. The FAQ questions are converted into TF-IDF vectors.
2. The user's question is converted into a vector.
3. Cosine similarity finds the closest stored question.
4. The app returns the corresponding answer if the similarity is high enough.

This is a retrieval-based FAQ bot, not a large language model or a fully generative AI chatbot. Add more question-answer pairs to expand it.
