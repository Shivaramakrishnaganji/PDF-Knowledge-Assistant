"""
This module handles breaking down large text documents into smaller, manageable chunks.

WHY THIS FILE EXISTS:
LLMs have a context window (a limit on how much text they can process at once). 
If we send an entire 100-page PDF to Gemini, it will fail or lose important details.
By splitting the text into chunks, we can search for only the most relevant pieces
and send those specific pieces to the LLM. 
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_logger
from src.constants import CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of LangChain Documents into smaller Document chunks.
    
    WHY THIS FUNCTION EXISTS:
    We use RecursiveCharacterTextSplitter because it tries to keep paragraphs and sentences
    together. It splits by double newline ('\n\n'), then single newline ('\n'), then space (' ').
    The overlap ensures that if a sentence is split across two chunks, the context isn't lost.
    
    Args:
        documents (List[Document]): The full-page documents extracted from the PDF.
        
    Returns:
        List[Document]: A new list of smaller Document chunks.
    """
    try:
        if not documents:
            logger.warning("No documents provided to split.")
            return []
            
        logger.info(f"Splitting {len(documents)} pages into smaller chunks...")
        
        # Initialize the splitter with our constants
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split the documents
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Successfully split into {len(chunks)} chunks.")
        return chunks
        
    except Exception as e:
        logger.error(f"Error occurred while splitting documents: {str(e)}")
        raise e
