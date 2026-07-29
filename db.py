import streamlit as st
import pandas as pd
import re
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Internship Support Chatbot", layout="centered")

st.title(" Internship Support Chatbot")
st.markdown("Ask me anything about tasks, deadlines, submissions, or technical issues!")

# ---- Load saved artifacts ----
faq_df = pd.read_csv("faq_data.csv")
vectorizer = joblib.load("vectorizer.pkl")
faq_vectors = joblib.load("faq_vectors.pkl")

CONFIDENCE_THRESHOLD = 0.25

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_response(user_query):
    clean_query = clean_text(user_query)
    query_vec = vectorizer.transform([clean_query])
    sims = cosine_similarity(query_vec, faq_vectors).flatten()
    best_idx = sims.argmax()
    best_score = sims[best_idx]

    if best_score < CONFIDENCE_THRESHOLD:
        return "I'm not fully sure about that. Please raise a support ticket or contact your mentor for help.", best_score
    return faq_df.iloc[best_idx]["answer"], best_score

# ---- Chat session state ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your Internship Support Assistant. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    answer, confidence = get_response(user_input)

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"Confidence score: {confidence:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown("---")
with st.expander(" View all FAQ topics I can help with"):
    st.dataframe(faq_df[["question"]])
