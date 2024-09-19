#!/bin/bash

# Start Ollama in the background
ollama start &

# Wait for a few seconds to ensure Ollama starts
sleep 5

# Check Ollama status
ollama status


sleep 5


#run ollama model llama3
ollama run llama3


# Start the Flask application
python app.py
