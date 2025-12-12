#!/bin/bash

# ================================
# CONFIG
# ================================
QnA="data/ground-truth/QA-Dataset.csv"
TIMESTAMP=2025-12-12_17-03-35

OUTPUT="./outputs/$TIMESTAMP"
mkdir -p "$OUTPUT"

METHOD="doublepass"   # semantic | recursive | doublepass
EMBEDDERS=("embeddinggemma" "all-minilm" "qwen3-embedding")

declare -A METHOD_PORT=(
  ["semantic"]=11435
  ["recursive"]=11436
  ["doublepass"]=11437
)

# ================================
# 1. Start Ollama
# ================================
echo "🚀 Starting Ollama for method: $METHOD"

port=${METHOD_PORT[$METHOD]}
session_ollama="ollama-$METHOD"

if tmux has-session -t "$session_ollama" 2>/dev/null; then
    echo "⚠️ Ollama session already running: $session_ollama"
else
    echo "🚀 Launching Ollama on port $port..."
    tmux new-session -d -s "$session_ollama" \
      "export OLLAMA_HOST=127.0.0.1:$port; ~/ollama/bin/ollama serve; read"
    sleep 5
fi

# ================================
# 2. Start Qdrant
# ================================
QDRANT_SESSION="qdrant"

if tmux has-session -t "$QDRANT_SESSION" 2>/dev/null; then
    echo "⚠️ Qdrant already running."
else
    echo "🚀 Starting Qdrant..."
    tmux new-session -d -s "$QDRANT_SESSION" "./qdrant; read"
    sleep 5
fi

# ================================
# 3. Run Evaluations (method × embedder)
# ================================
echo "🚀 Launching evaluation sessions..."

ollama_url="http://127.0.0.1:$port"

for embedder in "${EMBEDDERS[@]}"; do

    session="eval-$METHOD-$embedder"
    echo "  🔹 Starting: $session"

    tmux new-session -d -s "$session" \
        "python evaluate_method.py \
            $QnA \
            $METHOD \
            $embedder \
            $OUTPUT \
            $ollama_url \
            true; read"

done

echo "✅ Lite evaluation started!"
echo "👉 Output directory: $OUTPUT"
echo "👉 tmux ls   (cek semua session)"
