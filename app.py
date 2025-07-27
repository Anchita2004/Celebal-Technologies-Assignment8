import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import gradio as gr
from transformers import pipeline

# Load your dataset
df = pd.read_csv("loan_approval_dataset.csv")
df.columns = [col.strip() for col in df.columns]
df["text"] = df.apply(lambda row: " | ".join([f"{col}: {str(row[col])}" for col in df.columns]), axis=1)

# TF-IDF + FAISS indexing
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
dense_X = X.toarray().astype("float32")
index = faiss.IndexFlatL2(dense_X.shape[1])
index.add(dense_X)

# Load a free LLM (T5-based QA model)
llm = pipeline("text2text-generation", model="google/flan-t5-base")

# Function to generate answer
def generate_answer(question):
    question_vec = vectorizer.transform([question]).toarray().astype("float32")
    _, I = index.search(question_vec, k=3)
    context = "\n".join(df.iloc[i]["text"] for i in I[0])
    
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = llm(prompt, max_new_tokens=128)[0]["generated_text"]
    return response.strip()

# Gradio UI
gr.Interface(
    fn=generate_answer,
    inputs="text",
    outputs="text",
    title="Loan Dataset RAG Q&A Chatbot",
    description="Ask questions about the loan dataset. Example: 'What is the loan amount of male graduates?'"
).launch()
