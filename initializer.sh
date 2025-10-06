#!/bin/bash
# Install all requirements
pip install -r requirements.txt

# intall Ollama for evaluation
pip install -U langchain-ollama
# Install spaCy model (example: small English model)
python -m spacy download en_core_web_sm