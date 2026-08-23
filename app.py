import streamlit as st
import nltk
import numpy as np
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# Download NLTK Data
# ==========================================

@st.cache_resource
def download_nltk():

    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")


download_nltk()


# ==========================================
# Title
# ==========================================

st.title("🤖 AI FAQ Chatbot")

st.write(
    "Ask me anything about Artificial Intelligence!"
)

st.divider()


# ==========================================
# FAQ Dataset
# ==========================================

faq_data = {

    "What is Artificial Intelligence?":
    "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think, learn, and solve problems like humans.",

    "What is Machine Learning?":
    "Machine Learning is a subset of AI where systems learn from data and improve their performance over time without being explicitly programmed.",

    "What is Deep Learning?":
    "Deep Learning is a subset of Machine Learning that uses neural networks with many layers to analyze large amounts of data.",

    "What is a Neural Network?":
    "A Neural Network is a series of algorithms that mimic the human brain to recognize patterns and solve complex problems.",

    "What is Natural Language Processing?":
    "NLP is a branch of AI that helps computers understand, interpret, and generate human language.",

    "What is Computer Vision?":
    "Computer Vision is a field of AI that enables machines to interpret and understand visual information from images and videos.",

    "What is supervised learning?":
    "Supervised learning is a type of ML where the model is trained on labeled data — each input has a corresponding correct output.",

    "What is unsupervised learning?":
    "Unsupervised learning is where the model finds patterns in data without labeled responses.",

    "What is reinforcement learning?":
    "Reinforcement learning is a type of ML where an agent learns by interacting with its environment and receiving rewards or penalties.",

    "What is overfitting?":
    "Overfitting occurs when a model learns the training data too well, including noise, and performs poorly on new unseen data.",

    "What is a Large Language Model?":
    "A Large Language Model (LLM) is an AI model trained on massive amounts of text data to understand and generate human-like text.",

    "What is the difference between AI and ML?":
    "AI is the broad concept of machines being smart. ML is a specific technique used to achieve AI by training models on data.",

    "What are some real-world uses of AI?":
    "AI is used in healthcare (diagnosis), finance (fraud detection), self-driving cars, virtual assistants, recommendation systems, and much more.",

    "What is a chatbot?":
    "A chatbot is an AI-powered program that simulates conversation with users, typically to answer questions or provide support.",

    "What is the Turing Test?":
    "The Turing Test is a test proposed by Alan Turing to determine whether a machine can exhibit intelligent behavior indistinguishable from a human."
}


questions = list(faq_data.keys())
answers = list(faq_data.values())


# ==========================================
# Text Preprocessing
# ==========================================

stop_words = set(stopwords.words("english"))


def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# ==========================================
# TF-IDF Model
# ==========================================

@st.cache_resource
def create_model():

    cleaned_questions = [
        preprocess(question)
        for question in questions
    ]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(
        cleaned_questions
    )

    return vectorizer, tfidf_matrix


vectorizer, tfidf_matrix = create_model()


# ==========================================
# Find Best Answer
# ==========================================

def get_best_answer(user_input):

    if not user_input.strip():

        return "Please type a question."

    cleaned_input = preprocess(user_input)

    input_vector = vectorizer.transform(
        [cleaned_input]
    )

    similarities = cosine_similarity(
        input_vector,
        tfidf_matrix
    ).flatten()

    best_index = np.argmax(similarities)

    best_score = similarities[best_index]

    # Minimum similarity threshold
    if best_score < 0.1:

        return (
            "❌ Sorry, I couldn't find a matching answer. "
            "Please try rephrasing your question."
        )

    question = questions[best_index]

    answer = answers[best_index]

    return f"""
**Question:** {question}

**Answer:** {answer}

**Similarity Score:** {best_score:.2f}
"""


# ==========================================
# Chat History
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================
# Display Chat History
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# User Input
# ==========================================

user_input = st.chat_input(
    "Type your question here..."
)


# ==========================================
# Generate Answer
# ==========================================

if user_input:

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):

        st.markdown(user_input)


    # Get chatbot answer
    response = get_best_answer(user_input)


    # Add chatbot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })


    # Display chatbot answer
    with st.chat_message("assistant"):

        st.markdown(response)


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.header("🤖 AI FAQ Chatbot")

    st.write(
        "This chatbot answers frequently asked "
        "questions about Artificial Intelligence."
    )

    st.divider()

    st.subheader("Available Topics")

    st.write("• Artificial Intelligence")
    st.write("• Machine Learning")
    st.write("• Deep Learning")
    st.write("• Neural Networks")
    st.write("• NLP")
    st.write("• Computer Vision")
    st.write("• Reinforcement Learning")
    st.write("• LLM")
    st.write("• Chatbots")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()
