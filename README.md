Medical QA Chatbot

A Streamlit-based question-answering chatbot designed for academic and medical queries. The chatbot uses LangGraph, LangChain, and Groq API for context-aware answers, leveraging a vector store built with Chroma and HuggingFace embeddings.

🚀 Features

Contextual QA using LangGraph workflow.

Vector similarity search using Chroma and HuggingFace embeddings.

Integration with Groq API for LLM responses.

Display of source documents and page numbers for transparency.

Persistent chat session state using Streamlit session storage.

Scalable architecture for adding more data sources.

🗂 Project Structure
RAG/
│
├─ app.py                # Main Streamlit app
├─ config.py             # Configuration file (embedding model, directories, etc.)
├─ .env                  # Environment variables (GROQ_API_KEY)
├─ requirements.txt      # Project dependencies
├─ data/                 # PDF / document data for vectorization
├─ .venv/                # Python virtual environment
└─ README.md             # Project documentation

⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/<your-username>/medical-qa-chatbot.git
cd medical-qa-chatbot

2. Create a virtual environment
python -m venv .venv


Activate the virtual environment:

Windows (PowerShell)

.\.venv\Scripts\Activate


Linux / MacOS

source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt


Optional: Install hf_xet for faster HuggingFace model downloads:

pip install hf_xet

4. Set Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=<your_groq_api_key_here>


Important: Do not hardcode your API key in app.py.

5. Configure Embeddings and Vector Store

Open config.py and set:

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # For testing
CHROMA_PERSIST_DIRECTORY = "data/chroma"


You can switch to "intfloat/multilingual-e5-large" for full-scale deployment (requires 2.2GB download).

6. Run the Streamlit App
streamlit run app.py


Access the app in your browser: http://localhost:8504

📂 Adding Documents

Place your PDF or text documents in the data/ folder.

Use your Chroma vector store to index them:

# Chroma persist directory: data/chroma

⚡ Notes & Tips

First-time model download may take several minutes depending on your internet speed.

Windows users may see symlink warnings — these are safe to ignore.

For faster downloads on Windows:

Run Python as Administrator, or

Enable Developer Mode.

Session history is stored in Streamlit’s session state — closing the app resets the chat.

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

Add multi-document support with advanced context merging.

Enable real-time embeddings update for new documents.

Integrate voice input/output for a more interactive QA system.

Deploy on Streamlit Cloud / Docker / AWS for team access.
