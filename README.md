# 🧠 Loan Dataset RAG Q&A Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot built using a Loan Approval Dataset. It allows users to ask questions in natural language and get responses based on similar rows in the dataset, powered by an open-source LLM (`google/flan-t5-base`).

🔗 **Live Demo**: [Click here to try it on Hugging Face Spaces](https://huggingface.co/spaces/Anchita2004/loan-rag-chatbot1)

---

## 💡 Features

- ✅ **Fully free and open-source**
- 📄 Uses **Loan Approval CSV data**
- 🔍 TF-IDF for similarity search (via FAISS)
- 🧠 Answers generated using **Flan-T5 base model**
- 🧾 No API key or credits required (Hugging Face-hosted)

---

## 📁 Dataset

The dataset used is `loan_approval_dataset.csv`, which includes fields such as:

- ApplicantIncome
- LoanAmount
- Loan_Status
- Credit_History
- Education
- Self_Employed

Each row is converted into a searchable text chunk.

---

## 🚀 How It Works

1. Vectorizes the dataset using **TF-IDF**.
2. Stores embeddings in a **FAISS** index for fast similarity search.
3. When a question is asked:
   - Finds top 3 similar rows.
   - Passes them as context to a **Flan-T5 model**.
   - Generates an answer based on the context.

---

## 🛠 Tech Stack

- Python
- Pandas, Numpy, Scikit-learn
- FAISS for vector search
- Transformers (`google/flan-t5-base`)
- Gradio for web interface
- Hugging Face Spaces (deployment)

---

## 🧪 Example Questions

Try asking:

- "What is the loan status of self-employed applicants?"
- "Which applicant had the highest loan amount?"
- "Was credit history considered for loan approval?"

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
