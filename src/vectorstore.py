"""
This module handles the creation and querying of the vector database (FAISS).

WHY THIS FILE EXISTS:
Once we convert text chunks into numbers (embeddings), we need a fast way to store 
and search them. FAISS (Facebook AI Similarity Search) is an in-memory library 
optimized for exactly this. This module abstracts the FAISS logic away from the main app.
"""

from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from src.config import get_logger

logger = get_logger(__name__)

def create_vector_store(chunks: List[Document], embedding_model: Embeddings) -> FAISS:
    """
    Creates a FAISS vector database from a list of text chunks.
    
    WHY THIS FUNCTION EXISTS:
    This function takes the raw text chunks, passes them to the embedding model to get 
    the vectors, and then loads both the text and the vectors into FAISS. The resulting 
    'vector_store' object allows us to perform rapid similarity searches.
    
    Args:
        chunks (List[Document]): The text chunks to store.
        embedding_model (Embeddings): The model used to convert text to vectors.
        
    Returns:
        FAISS: The initialized vector store containing the chunks.
    """
    try:
        if not chunks:
            raise ValueError("Cannot create a vector store from an empty list of chunks.")
            
        logger.info(f"Creating FAISS vector store for {len(chunks)} chunks...")
        
        # FAISS.from_documents automatically calculates embeddings for each chunk 
        # and builds the index in memory.
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
        
        logger.info("Successfully created FAISS vector store.")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise e
