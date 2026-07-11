"""
This module orchestrates the core RAG (Retrieval-Augmented Generation) logic.

WHY THIS FILE EXISTS:
Instead of using a black-box function like `create_retrieval_chain`, we implement the 
pipeline explicitly here. This makes it highly readable and interview-friendly. 
We retrieve chunks, format them, build a prompt, call the LLM, and parse the response.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from openai import NotFoundError

from src.prompts import get_rag_prompt
from src.config import get_logger, OPENAI_API_KEY
from src.constants import DEFAULT_TOP_K

logger = get_logger(__name__)

def generate_answer(
    question: str, 
    vector_store: FAISS, 
    top_k: int = DEFAULT_TOP_K
) -> Tuple[str, List[Document]]:
    """
    Executes the explicit RAG pipeline to generate an answer.
    
    WHY THIS FUNCTION EXISTS:
    This is the heart of the application. It breaks down the RAG process into explicit, 
    understandable steps: Retrieval -> Formatting -> Prompt Creation -> LLM Call.
    
    Args:
        question (str): The user's question.
        vector_store (FAISS): The populated FAISS database.
        top_k (int): How many chunks to retrieve.
        
    Returns:
        Tuple[str, List[Document]]: A tuple containing the text answer and the source documents used.
    """
    try:
        logger.info(f"Generating answer for question: '{question}' (Top K: {top_k})")
        
        # Step 1: Retrieval
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        source_documents = retriever.invoke(question)
        
        if not source_documents:
            logger.warning("No relevant context found in the document.")
            return "No relevant context found in the document to answer your question.", []
            
        # Step 2: Format the Context
        context_text = "\n\n---\n\n".join([doc.page_content for doc in source_documents])
        
        # Step 3: Build the Prompt
        prompt_template = get_rag_prompt()
        formatted_prompt = prompt_template.format(context=context_text, question=question)
        
        # Step 4 & 5: Initialize the LLM and Call it
        # We try gpt-4.1-mini first, and fallback to gpt-4o-mini if it throws a NotFoundError.
        try:
            logger.info("Attempting LLM invocation with gpt-4.1-mini...")
            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                openai_api_key=OPENAI_API_KEY,
                temperature=0.0
            )
            response = llm.invoke(formatted_prompt)
        except Exception as e:
            # Check if the error is due to the model not existing (404/NotFoundError)
            if "model" in str(e).lower() and ("does not exist" in str(e).lower() or "not found" in str(e).lower() or "404" in str(e).lower()):
                logger.warning("gpt-4.1-mini not found. Falling back to gpt-4o-mini...")
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=OPENAI_API_KEY,
                    temperature=0.0
                )
                response = llm.invoke(formatted_prompt)
            else:
                # If it failed for another reason (like bad API key), raise the error.
                raise e
        
        # Step 6: Return the Answer and the Sources
        logger.info("Successfully generated answer.")
        return response.content, source_documents
        
    except Exception as e:
        logger.error(f"Error during RAG generation: {str(e)}")
        raise e
