"""
This module handles the creation of the embedding model.

WHY THIS FILE EXISTS:
To perform a 'similarity search', we must convert text chunks into arrays of numbers 
(vectors/embeddings). Words with similar meanings will have numbers that are mathematically 
close to each other. This file initializes the Google model that performs this translation.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import get_logger, GOOGLE_API_KEY

logger = get_logger(__name__)

def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Generative AI Embeddings model.
    
    WHY THIS FUNCTION EXISTS:
    By isolating the model initialization here, we can easily swap it out later 
    (e.g., if we decide to use OpenAI or HuggingFace embeddings instead) without 
    changing the rest of our application code.
    
    Returns:
        GoogleGenerativeAIEmbeddings: The initialized embedding model.
    """
    try:
        logger.info("Initializing Google Generative AI Embeddings model...")
        
        # We use the text-embedding-004 model, which is Google's standard embedding model.
        # It requires the API key to authenticate with Google's servers.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}")
        raise e
