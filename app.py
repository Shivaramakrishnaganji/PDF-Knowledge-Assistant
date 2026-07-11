"""
This is the main entry point for the Streamlit web application.

WHY THIS FILE EXISTS:
This file handles the presentation layer (User Interface). It manages user inputs,
maintains session state (like chat history and vector store), and stitches together
the logic from the `src/` modules. By keeping logic in `src/` and UI here, the code 
is clean and modular.
"""

import os
import tempfile
import streamlit as st

# Import our custom modules
from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vectorstore import create_vector_store
from src.rag import generate_answer
from src.constants import APP_TITLE, APP_DESCRIPTION, DEFAULT_TOP_K

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="centered")

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
# Streamlit reruns the script on every interaction. We use session_state to remember things.
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores chat history

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # Stores the FAISS database in memory

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def process_uploaded_file(uploaded_file):
    """Handles the entire pipeline from file upload to vector store creation."""
    try:
        # Step 1: Save the uploaded file to a temporary location on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with st.spinner("Loading PDF..."):
            documents = load_pdf(tmp_path)
            
        with st.spinner("Chunking text..."):
            chunks = split_documents(documents)
            
        with st.spinner("Generating embeddings and building vector store..."):
            embedding_model = get_embedding_model()
            vector_store = create_vector_store(chunks, embedding_model)
            
            # Save to session state so it persists across UI interactions
            st.session_state.vector_store = vector_store
            
        # Clean up the temporary file
        os.remove(tmp_path)
        
        st.success(f"Successfully processed **{uploaded_file.name}**!")
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"An error occurred while processing the PDF: {error_msg}")
        
        # Diagnostic check for 404 Embedding Model errors
        if "404" in error_msg and "is not found" in error_msg:
            try:
                import google.generativeai as genai
                from src.config import GOOGLE_API_KEY
                genai.configure(api_key=GOOGLE_API_KEY)
                
                # Fetch available models
                available_models = []
                for m in genai.list_models():
                    if 'embedContent' in m.supported_generation_methods:
                        available_models.append(m.name)
                        
                if available_models:
                    st.info(f"**Diagnostic Info:** Your API key supports these embedding models:\n\n`{', '.join(available_models)}`")
                    st.warning("Please let Antigravity know which models are listed above so we can update the code!")
                else:
                    st.error("**Diagnostic Info:** Your API key does not have access to ANY embedding models.")
            except Exception as diag_e:
                st.error(f"Could not run diagnostic check: {str(diag_e)}")


# -----------------------------------------------------------------------------
# UI Layout: Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    # Process Button
    if uploaded_file and st.button("Process Document"):
        process_uploaded_file(uploaded_file)
        
    st.divider()
    
    # Retrieval Settings
    st.subheader("Retrieval Options")
    top_k = st.slider("Number of source chunks (Top K)", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    
    st.divider()
    
    # Clear Chat Button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# UI Layout: Main Page
# -----------------------------------------------------------------------------
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If the AI provided source chunks, display them in an expander
        if "sources" in message and message["sources"]:
            with st.expander("View Source Chunks"):
                for i, doc in enumerate(message["sources"]):
                    page = doc.metadata.get("page", "Unknown")
                    st.markdown(f"**Chunk {i+1} (Page {page}):**")
                    st.info(doc.page_content)

# -----------------------------------------------------------------------------
# Chat Input
# -----------------------------------------------------------------------------
# Accept user input at the bottom of the screen
if prompt := st.chat_input("Ask a question about the PDF..."):
    
    # Check if a document has been processed
    if not st.session_state.vector_store:
        st.error("Please upload and process a PDF document first.")
    else:
        # 1. Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 2. Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call our explicit RAG function
                    answer, source_docs = generate_answer(
                        question=prompt, 
                        vector_store=st.session_state.vector_store,
                        top_k=top_k
                    )
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Display the sources in an expander
                    if source_docs:
                        with st.expander("View Source Chunks"):
                            for i, doc in enumerate(source_docs):
                                page = doc.metadata.get("page", "Unknown")
                                st.markdown(f"**Chunk {i+1} (Page {page}):**")
                                st.info(doc.page_content)
                                
                    # Save AI response to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": source_docs
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
