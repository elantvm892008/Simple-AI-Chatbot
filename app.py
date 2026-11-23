import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Simple AI Chatbot", page_icon="🤖")
st.title("🤖 Simple AI Chatbot")
st.write("Ask a few beginner-friendly questions about Python, AIML and this demo.")

# Small FAQ knowledge base. Add more question-answer pairs to improve the bot.
questions = [
    "What is Python?",
    "What is machine learning?",
    "What is artificial intelligence?",
    "What is AIML?",
    "What is Pandas?",
    "What is NumPy?",
    "What is Scikit-learn?",
    "What is GitHub?",
    "What is a dataset?",
    "What is supervised learning?",
    "What is NLP?",
    "How do I learn Python?"
]

answers = [
    "Python is a popular, beginner-friendly programming language used in web development, automation, data science and AI.",
    "Machine learning is a field of AI where computers learn patterns from data to make predictions or decisions.",
    "Artificial intelligence is the broader field of building computer systems that perform tasks associated with human intelligence.",
    "AIML stands for Artificial Intelligence and Machine Learning.",
    "Pandas is a Python library used to load, clean, analyse and work with tabular data.",
    "NumPy is a Python library for efficient numerical operations and arrays.",
    "Scikit-learn is a Python library that provides tools for common machine-learning tasks.",
    "GitHub is a platform for hosting Git repositories and collaborating on software projects.",
    "A dataset is a collection of related data used for analysis or to train and test machine-learning models.",
    "Supervised learning trains a model using examples that include both input data and the correct answer or label.",
    "NLP stands for Natural Language Processing. It helps computers work with human language.",
    "Start with variables, data types, conditions, loops, functions and lists. Then build small projects and practise regularly."
]

@st.cache_resource
def build_search_model():
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix

vectorizer, question_matrix = build_search_model()

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "content": "Hello! Ask me something about Python or AIML."}
    ]

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Type your question...")
if user_question:
    st.session_state.chat.append({"role": "user", "content": user_question})

    user_vector = vectorizer.transform([user_question])
    scores = cosine_similarity(user_vector, question_matrix)[0]
    best_index = int(scores.argmax())

    if scores[best_index] >= 0.18:
        reply = answers[best_index]
    else:
        reply = "I don't know that yet. Try asking about Python, machine learning, AIML, Pandas, NumPy or GitHub."

    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()

with st.sidebar:
    st.subheader("About this chatbot")
    st.write("This is a small FAQ chatbot. It matches your question to the closest stored question using TF-IDF and cosine similarity. It is not a generative AI model.")
    if st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()
