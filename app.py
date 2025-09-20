import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangGraph and Langchain imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config

# --- Load Environment Variables ---
load_dotenv()

# Get Groq API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Please set it in .env or system environment.")

# --- Configuration and Initialization ---

# Initialize embeddings model
def get_embeddings_model():
    """Caches the HuggingFaceEmbeddings model."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

embeddings = get_embeddings_model()

# Initialize Chroma vector store
def get_vector_store(embed_func):
    """Caches the Chroma vector store."""
    try:
        return Chroma(
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            embedding_function=embed_func
        )
    except Exception as e:
        st.error(f"Error loading ChromaDB. Make sure '{config.CHROMA_PERSIST_DIRECTORY}' exists and is populated. Error: {e}")
        st.stop()  # Stop the app if DB cannot be loaded

vectordb = get_vector_store(embeddings)

# Initialize the ChatGroq model
def get_chat_model():
    """Caches the ChatGroq model."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,  # Lower temperature for more consistent answers
        max_tokens=400,
        api_key=api_key   # ✅ Explicitly pass API key
    )

model = get_chat_model()

# Define the LangGraph node function
def call_model(state: MessagesState):
    """
    This function defines the 'model' node in the LangGraph workflow.
    It takes the current state (conversation messages) and invokes the LLM.
    """
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Answer all questions to the best of your ability."
    )

    # Prepend the system message to the current conversation history
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}

# Build and compile the LangGraph workflow
def get_langgraph_app():
    """Caches and compiles the LangGraph workflow."""
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    # Add simple in-memory checkpointer for conversation history
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

app = get_langgraph_app()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Academic QA", layout="centered")
st.title("💬 Medical QA")

# Initialize session state for messages and thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores all messages for display
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_chat_session"  # Fixed session ID

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Logic ---
if prompt := st.chat_input("Ask any question..."):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Retrieve context for the current user question
                docs = vectordb.similarity_search_with_score(prompt, k=3)
                _docs = pd.DataFrame(
                    [(prompt, doc[0].page_content, doc[0].metadata.get('source'), doc[0].metadata.get('page'), doc[1]) for doc in docs],
                    columns=['query', 'paragraph', 'document', 'page_number', 'relevant_score']
                )
                current_context = "\n\n".join(_docs['paragraph'])

                # 2. Construct the HumanMessage for the current turn, including context
                current_turn_message = HumanMessage(content=f"Context: {current_context}\n\nQuestion: {prompt}")

                # 3. Invoke the LangGraph app
                result = app.invoke(
                    {"messages": [current_turn_message]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )

                # Get the AI's response (the last AIMessage in the result)
                ai_response = result['messages'][-1].content

                # Extract source document and page numbers
                source_document = _docs['document'][0] if not _docs.empty and 'document' in _docs.columns else "N/A"
                top_three_page_numbers = _docs['page_number'].drop_duplicates().head(3).astype(str).tolist()
                page_numbers_str = ', '.join(top_three_page_numbers) if top_three_page_numbers else "N/A"

                # Format the final response
                final_response = f"{ai_response}\n\n**Source Document**: {source_document}\n**Reference Page Numbers**: {page_numbers_str}"
                
                st.markdown(final_response)

                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": "I encountered an error. Please try again."}
                )

# To run:
# 1. Save this code as `app.py`.
# 2. Create a `.env` file in your project root with:
#    GROQ_API_KEY=your_api_key_here
# 3. Run: `streamlit run app.py`
