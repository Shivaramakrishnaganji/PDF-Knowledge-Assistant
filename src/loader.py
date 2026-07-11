"""
This module handles loading and extracting text from PDF files.

WHY THIS FILE EXISTS:
Before we can search through a PDF or send its content to an LLM, we must convert 
the binary PDF file into plain text. This module takes a temporary file path, 
extracts the text page by page, and returns LangChain Document objects.
"""

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from src.config import get_logger

logger = get_logger(__name__)

def load_pdf(file_path: str) -> List[Document]:
    """
    Loads a PDF from a given file path and extracts its pages as Document objects.
    
    WHY THIS FUNCTION EXISTS:
    LangChain's text splitters and vector stores expect data in a specific format 
    called a 'Document' (which contains 'page_content' and 'metadata'). This function
    bridges the gap between a raw PDF file and LangChain's required format.
    
    Args:
        file_path (str): The absolute or relative path to the PDF file.
        
    Returns:
        List[Document]: A list of LangChain Document objects, typically one per page.
        
    Raises:
        Exception: If the file cannot be loaded or is corrupted.
    """
    try:
        logger.info(f"Starting to load PDF from path: {file_path}")
        
        # Initialize the PyPDFLoader with the file path
        loader = PyPDFLoader(file_path)
        
        # Load the document. This reads the PDF and splits it into pages automatically.
        documents = loader.load()
        
        # Check if the PDF was empty (no text extracted)
        if not documents:
            logger.warning(f"No text found in the PDF: {file_path}")
            return []
            
        logger.info(f"Successfully loaded {len(documents)} pages from the PDF.")
        return documents
        
    except Exception as e:
        logger.error(f"Error occurred while loading PDF '{file_path}': {str(e)}")
        # Re-raise the exception so the caller (Streamlit app) can show an error to the user
        raise e
