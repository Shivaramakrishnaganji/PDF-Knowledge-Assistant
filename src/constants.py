"""
This module centralizes all magic numbers and constant values.

WHY THIS FILE EXISTS:
Hardcoding numbers like 700 (chunk size) inside functions makes the code hard to read and 
hard to tune. By putting them here, a developer can quickly adjust the behavior of the 
entire application from one place without searching through multiple files.
"""

# Text Splitting Constants
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# Retrieval Constants
DEFAULT_TOP_K = 4

# UI Constants
APP_TITLE = "PDF Knowledge Assistant"
APP_DESCRIPTION = "Upload a PDF and ask questions about its content using RAG."
