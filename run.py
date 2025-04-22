"""
Main entry point for the e-DocInsight application.
"""
import sys
from pathlib import Path
from doc_retriever.api.app import main

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

if __name__ == "__main__":
    main() 