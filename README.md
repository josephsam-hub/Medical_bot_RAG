# Medical QA Chatbot

![Streamlit](https://img.shields.io/badge/Streamlit-App-blue) 
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

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

![Demo GIF](https://via.placeholder.com/600x300.png?text=Demo+Coming+Soon)

> Replace the placeholder GIF with an actual screen recording of your app.

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

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/medical-qa-chatbot.git
cd medical-qa-chatbot
2. Create and activate a virtual environment
Windows (PowerShell):

powershell
Copy code
python -m venv .venv
.\.venv\Scripts\Activate
Linux / MacOS:

bash
Copy code
python -m venv .venv
source .venv/bin/activate
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
Optional for faster downloads:

bash
Copy code
pip install hf_xet
4. Set Environment Variables
Create a .env file in the project root:

env
Copy code
GROQ_API_KEY=<your_groq_api_key_here>
Do not hardcode your API key in app.py.

5. Configure Embeddings and Vector Store
In config.py:

python
Copy code
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Light model for testing
CHROMA_PERSIST_DIRECTORY = "data/chroma"
For production, you can use "intfloat/multilingual-e5-large" (requires 2.2GB download).

6. Run the Streamlit App
bash
Copy code
streamlit run app.py
Access the app in your browser:

arduino
Copy code
http://localhost:8504
📂 Adding Documents
Place your PDF or text files in the data/ folder.

The Chroma vector store will automatically index them.

⚡ Notes & Tips
First-time model download may take several minutes.

Windows users may see symlink warnings — safe to ignore.

For faster downloads on Windows:

Run Python as Administrator, or

Enable Developer Mode.

Session history is stored in Streamlit’s session state — closing the app resets chat.

🛠 Tech Stack
Component	Library / Tool
Streamlit UI	streamlit
LLM API	langchain_groq
Workflow Engine	langgraph
Vector Store	Chroma
Embeddings	HuggingFaceEmbeddings
Environment Mgmt	python-dotenv
Data Processing	pandas

📈 Future Improvements
Multi-document support with advanced context merging

Real-time embeddings update for new documents

Voice input/output for interactive QA

Deploy on Streamlit Cloud / Docker / AWS
