"""
This module handles centralized configuration and logging setup.

WHY THIS FILE EXISTS:
In a production application, you should never hardcode configuration values
(like API keys) across multiple files. By centralizing them here, we ensure:
1. Environment variables are loaded exactly once.
2. Logging is configured consistently across the entire application.
3. If a key is missing, the application fails fast and early with a clear message.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
# This allows us to keep secrets (like API keys) out of source control.
load_dotenv()

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a standard logger.
    
    WHY THIS FUNCTION EXISTS:
    Using print() statements is bad practice in production. A proper logger allows us 
    to filter messages by severity (INFO, WARNING, ERROR) and easily format output 
    (e.g., adding timestamps and module names).
    """
    logger = logging.getLogger(name)
    
    # Only configure the logger if it hasn't been configured yet to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler and set format
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        
    return logger

# Fetch the Google API key from environment variables.
# If it's missing, we raise an error immediately so the developer knows what to fix.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger = get_logger(__name__)
    logger.warning("GOOGLE_API_KEY is not set in the environment variables. The application may fail.")
