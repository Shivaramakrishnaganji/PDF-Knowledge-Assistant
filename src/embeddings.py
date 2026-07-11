"""
This module handles the creation of the embedding model.

WHY THIS FILE EXISTS:
To perform a 'similarity search', we must convert text chunks into arrays of numbers 
(vectors/embeddings). Words with similar meanings will have numbers that are mathematically 
close to each other. This file initializes the OpenAI model that performs this translation.
"""

from langchain_openai import OpenAIEmbeddings
from src.config import get_logger, OPENAI_API_KEY

logger = get_logger(__name__)

def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initializes and returns the OpenAI Embeddings model.
    
    WHY THIS FUNCTION EXISTS:
    By isolating the model initialization here, we can easily swap it out later 
    (e.g., if we decide to use HuggingFace embeddings instead) without 
    changing the rest of our application code.
    
    Returns:
        OpenAIEmbeddings: The initialized embedding model.
    """
    try:
        logger.info("Initializing OpenAI Embeddings model...")
        
        # We use text-embedding-3-small, which is OpenAI's fast, cheap, and highly capable model.
        # It requires the API key to authenticate with OpenAI's servers.
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}")
        raise e
