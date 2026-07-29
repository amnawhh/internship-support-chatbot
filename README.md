#  Internship Support Chatbot using NLP (TF-IDF + Cosine Similarity)

##  Objective
This project builds an AI-powered chatbot that answers common intern queries and provides real-time support, automating responses that would otherwise require a mentor or coordinator. Built as Task 6 of my Machine Learning Internship at Internee.pk.

##  Dataset
Since real FAQ documents and support ticket logs aren't publicly available, this project uses a synthetically generated dataset that realistically simulates:
- **FAQ Documents**: 15 common intern questions and answers covering submissions, deadlines, certificates, mentor contact, and technical issues.
- **Historical Support Tickets**: 60 simulated tickets across 6 categories (Submission Issue, Deadline Extension, Technical Error, Certificate Query, Mentor Contact, Tool Setup) with resolution status.

##  Methodology
1. **Text Preprocessing** – Cleaned and normalized all FAQ questions and user queries.
2. **TF-IDF Vectorization** – Converted the FAQ question bank into numerical vectors.
3. **Retrieval-Based Matching** – For any incoming user query, computed cosine similarity against the entire FAQ bank and returned the answer to the closest-matching question.
4. **Confidence Thresholding** – If the best match's similarity score falls below 0.25, the chatbot returns a fallback response directing the intern to raise a support ticket, instead of guessing.
5. **Evaluation** – Sanity-checked the chatbot by feeding it the FAQ questions themselves and measuring self-match accuracy.
6. **Support Ticket Analysis** – Visualized ticket volume and resolution rate by category to understand where interns need the most help.

> **Note on model choice:** The task guidelines suggested Rasa or Hugging Face Transformers. Due to the same Python 3.13 / PyTorch compatibility issues encountered in Task 4, I implemented a lightweight **TF-IDF + cosine similarity retrieval chatbot** instead. This approach still satisfies the NLP-based automated response objective, is fully reproducible, and avoids environment-dependent installation failures — a practical trade-off documented for transparency.

##  Tech Stack
- Python
- pandas, numpy
- scikit-learn (TF-IDF, cosine similarity)
- matplotlib, seaborn
- Streamlit (chat interface)
- joblib (model persistence)

##  How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/internship-support-chatbot.git
cd internship-support-chatbot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook** (to regenerate data, vectors, and charts)
```bash
jupyter notebook Task6_Support_Chatbot.ipynb
```

**4. Launch the chatbot dashboard**
```bash
streamlit run app.py
```

##  Live Demo
 [View the deployed chatbot dashboard](https://internship-support-chatbot-5c5bzoyqzfrfxltfdikdd6.streamlit.app/)

