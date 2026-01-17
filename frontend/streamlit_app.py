"""
Streamlit App - Main entry point for the Leaf Disease Prediction application

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.frontend import run

if __name__ == "__main__":
    run()
