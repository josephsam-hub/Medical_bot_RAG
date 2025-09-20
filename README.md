## Medical QA Chatbot

![Streamlit](https://img.shields.io/badge/Streamlit-App-blue) 
![Python](https://img.shields.io/badge/Python-3.10+-yellow)


A Streamlit-based **question-answering chatbot** for academic and medical queries.  
It uses **LangGraph**, **LangChain**, and **Groq API** for context-aware answers, leveraging a vector store built with **Chroma** and **HuggingFace embeddings**.

---

## 🚀 Features

- Contextual QA using **LangGraph workflow**  
- Vector similarity search using **Chroma** and **HuggingFace embeddings**  
- Integration with **Groq API** for LLM responses  
- Displays **source documents and page numbers**  
- Persistent **chat session** using Streamlit session state  
- Scalable architecture for additional data sources  

---

## 🎬 Demo

![WhatsApp Image 2025-09-20 at 19 45 12_e9c113b5](https://github.com/user-attachments/assets/1814b91f-c7a1-4748-a44e-68794837e636)


---

## 🗂 Project Structure

RAG/
│
├─ app.py # Main Streamlit app

├─ config.py # Configuration (embedding model, directories)

├─ .env # Environment variables (GROQ_API_KEY)

├─ requirements.txt # Project dependencies

├─ data/ # PDF/text documents for vectorization

├─ .venv/ # Python virtual environment

└─ README.md # Project documentation


---

## ⚙️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/<your-username>/medical-qa-chatbot.git
cd medical-qa-chatbot

### 2. Create and activate a virtual environment
**Windows (PowerShell):**
python -m venv .venv
.\.venv\Scripts\Activate
**Linux / MacOS:**
python -m venv .venv
source .venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt
Optional for faster downloads:
pip install hf_xet

### 4. Set Environment Variables
Create a .env file in the project root:
GROQ_API_KEY=<your_groq_api_key_here>
Do not hardcode your API key in app.py.

### 5. Configure Embeddings and Vector Store
In config.py:

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Light model for testing
CHROMA_PERSIST_DIRECTORY = "data/chroma"
For production, you can use "intfloat/multilingual-e5-large" (requires 2.2GB download).

### 6. Run the Streamlit App

streamlit run app.py
Access the app in your browser:
http://localhost:8504

## 🛠 Tech Stack

| Component        | Library / Tool           |
|-----------------|-------------------------|
| Streamlit UI     | `streamlit`             |
| LLM API          | `langchain_groq`        |
| Workflow Engine  | `langgraph`             |
| Vector Store     | `Chroma`                |
| Embeddings       | `HuggingFaceEmbeddings` |
| Environment Mgmt | `python-dotenv`         |
| Data Processing  | `pandas`                |


###  📈 Future Improvements
Multi-document support with advanced context merging

Real-time embeddings update for new documents

Voice input/output for interactive QA

Deploy on Streamlit Cloud / Docker / AWS

