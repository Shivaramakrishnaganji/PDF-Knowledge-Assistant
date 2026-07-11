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
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompts import get_rag_prompt
from src.config import get_logger, GOOGLE_API_KEY
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
        # We query the vector store to find the chunks most mathematically similar to the question.
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        source_documents = retriever.invoke(question)
        
        if not source_documents:
            logger.warning("No relevant context found in the document.")
            return "No relevant context found in the document to answer your question.", []
            
        # Step 2: Format the Context
        # We extract the text from the retrieved Document objects and join them into a single string.
        # We separate chunks with newlines so the LLM knows they are distinct pieces of text.
        context_text = "\n\n---\n\n".join([doc.page_content for doc in source_documents])
        
        # Step 3: Build the Prompt
        # We inject the formatted context and the user's question into our PromptTemplate.
        prompt_template = get_rag_prompt()
        formatted_prompt = prompt_template.format(context=context_text, question=question)
        
        # Step 4: Initialize the LLM
        # We use Gemini 1.5 Flash, which is universally available and extremely fast.
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0  # Temperature 0 means highly deterministic (less creative, more factual)
        )
        
        # Step 5: Call the LLM
        # We send the fully constructed string to Gemini.
        response = llm.invoke(formatted_prompt)
        
        # Step 6: Return the Answer and the Sources
        # We return the content of the AI's message, plus the raw documents so the UI can display them.
        logger.info("Successfully generated answer.")
        return response.content, source_documents
        
    except Exception as e:
        logger.error(f"Error during RAG generation: {str(e)}")
        raise e
