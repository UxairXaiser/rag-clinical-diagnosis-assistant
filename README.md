---

# 🧠 RAG Clinical Diagnosis Assistant  
*A Retrieval-Augmented Generation system for clinical reasoning using MIMIC-IV-Ext DiReCT dataset.*

---

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that assists in clinical diagnostic reasoning by leveraging annotated clinical notes and structured medical knowledge. Built using the **MIMIC-IV-Ext DiReCT dataset**, it combines semantic search with generative language models to answer medical queries and generate informative diagnostic summaries.

The application features:
- 🔍 Dense retrieval of relevant clinical notes
- 🧾 Summarization and diagnostic generation via LLMs
- 🖥️ Interactive frontend built with **Streamlit**
- 🧠 Open-source transformer models from Hugging Face

---

## 📁 Project Structure

```
├── data/
│   ├── diagnostic_kg/            # Medical knowledge graphs (JSON)
│   └── samples/                  # Annotated clinical notes
├── notebook/
│   └── rag-assignment.ipynb      # Full exploratory notebook
├── app/
│   └── streamlit_app.py          # Streamlit UI code
├── utils/
│   └── retriever.py              # FAISS indexing & query logic
│   └── generator.py              # LLM-based generation
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 How It Works

### 🔎 Step 1: Dense Document Retrieval

We use the `sentence-transformers` model `all-MiniLM-L6-v2` to embed clinical documents. Then we use **FAISS** to index and retrieve the top-k most similar notes based on a user's medical query.

### 🧠 Step 2: Generative Answering

Using a transformer-based model (e.g., `google/flan-t5-base`), the system takes the retrieved documents and user query as context and generates a diagnostic summary.

### 🖼️ Step 3: Streamlit Frontend

The Streamlit app allows users to:
- Input natural language queries (e.g., "fever, fatigue, jaundice")
- View relevant clinical documents
- Get AI-generated diagnostic insights

---

## 🧪 Sample Query

```
Input Query:
"Patient has persistent cough, fever, and shortness of breath."

Output Diagnosis (Generated):
"The symptoms are suggestive of pneumonia. Recommended steps include chest imaging and initiation of antibiotic therapy..."
```

---

## 🧰 Installation

1. Clone the repo:

```bash
git clone https://github.com/uzairqaiser/rag-clinical-diagnosis-assistant.git
cd rag-clinical-diagnosis-assistant
```

2. Create a virtual environment & install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

---

## 📦 Requirements

Main libraries used:
- `sentence-transformers`
- `faiss-cpu`
- `transformers`
- `torch`
- `streamlit`
- `scikit-learn`

You can install everything via:

```bash
pip install -r requirements.txt
```

---

## 📈 Evaluation

We evaluate:
- **Retrieval quality** using manual Precision@K
- **Generation** based on coherence, factuality, and relevance

Although lightweight models were used for this prototype, results showed promising diagnostic relevance, especially with well-structured prompts.

---

## 🗂 Dataset

This project uses the [MIMIC-IV-Ext DiReCT](https://physionet.org/content/mimic-iv/) dataset (available through Google Classroom for this assignment), which contains:
- Structured medical knowledge graphs
- Annotated clinical notes categorized by disease

⚠️ **Note**: Please ensure compliance with ethical and privacy guidelines when handling clinical data.

---

## 📄 Blog & Documentation

- 📖 **Medium Blog Post**: [Read here](https://medium.com/@uzairqaiser/comingsoon)
- 🔗 **LinkedIn Post**: [Connect & view](https://linkedin.com/in/uzair-qaiser)
- 📚 **Jupyter Notebook**: [`rag-assignment.ipynb`](notebook/rag-assignment.ipynb)

---

## 📌 Future Improvements

- Integrate larger models like `Mistral-7B` or `LLaMA` for better generation
- Enhance evaluation with human expert feedback
- Deploy as a public web demo with user authentication

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

- PhysioNet & MIMIC-IV team for the dataset
- Hugging Face Transformers & Sentence Transformers libraries
- FAISS by Facebook AI Research

---
