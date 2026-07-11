"""
This module contains the prompts used to instruct the LLM.

WHY THIS FILE EXISTS:
Prompts are the steering wheel for an LLM. By keeping them in a separate file, 
we can easily tweak the AI's persona, rules, and constraints without digging 
through application logic.
"""

from langchain_core.prompts import PromptTemplate

# This prompt strictly tells the LLM to use ONLY the provided context.
# If the answer is not in the context, it must say so. This prevents hallucinations.
RAG_SYSTEM_PROMPT = """
You are an intelligent and helpful Knowledge Assistant. 
Your primary task is to answer the user's question based strictly on the provided context below.

Rules:
1. If the answer is not contained within the provided context, clearly state: "I don't know based on the provided document."
2. Do not hallucinate or make up information outside of the context.
3. Provide a clear, concise, and structured answer. Use bullet points if necessary.
4. If the context is ambiguous, explain the ambiguity based only on the text.

Context:
{context}

Question:
{question}

Answer:
"""

def get_rag_prompt() -> PromptTemplate:
    """
    Returns the PromptTemplate object for the RAG pipeline.
    
    WHY THIS FUNCTION EXISTS:
    LangChain uses PromptTemplate objects to dynamically inject variables (like {context} 
    and {question}) into the string at runtime.
    """
    return PromptTemplate(
        template=RAG_SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )
