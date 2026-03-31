#!/usr/bin/env python3
"""Startup script for the RAG Streamlit app with proper environment setup."""

import os
import subprocess
import sys
import warnings

def setup_environment():
    """Set up environment variables and warning filters."""
    # Suppress transformers warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    
    # Set environment variables
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
    
    print("Environment configured for RAG app startup")
    print("Starting Streamlit app...")

def main():
    """Start the Streamlit application."""
    setup_environment()
    
    try:
        # Run streamlit with the app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nApp stopped by user")
        return 0

if __name__ == "__main__":
    sys.exit(main())
