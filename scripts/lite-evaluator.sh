#!/bin/bash

# ================================
# CONFIG
# ================================
QnA="data/ground-truth/QA-Dataset.csv"
TIMESTAMP=2025-12-12_14-04-57

# Custom output directory (set custom name)
OUTPUT="./outputs/$TIMESTAMP"
mkdir -p "$OUTPUT"

# Choose only ONE method
METHOD="recursive"   # <-- ubah ke: semantic | recursive | doublepass

# Available embedder models
EMBEDDERS=("embeddinggemma" "all-minilm" "qwen3-embedding")

# Output formats
declare -A FORMATS=(
  [0]="1 paragraf ringkas padat jelas"
  [1]="poin utama dan penjelasan singkat"
  [2]="jawaban dalam bentuk tabel teks"
)

# Dedicated port for each method
declare -A METHOD_PORT=(
  ["semantic"]=11435
  ["recursive"]=11436
  ["doublepass"]=11437
)

# ================================
# 1. Start Ollama Server (only 1)
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
# 2. Start Qdrant (only 1)
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
# 3. Run evaluations 
#    (method × embedder × format)
# ================================
echo "🚀 Launching evaluation sessions..."

ollama_url="http://127.0.0.1:$port"

for embedder in "${EMBEDDERS[@]}"; do
    for format_index in "${!FORMATS[@]}"; do
        
        format="${FORMATS[$format_index]}"
        session="eval-$METHOD-$embedder-$format_index"

        echo "  🔹 Starting: $session"

        tmux new-session -d -s "$session" \
            "python evaluate_method.py \
                $QnA \
                $METHOD \
                $embedder \
                $format_index \
                \"$format\" \
                $OUTPUT \
                $ollama_url \
                true; read"
    done
done

echo "✅ Lite evaluation started!"
echo "👉 Output directory: $OUTPUT"
echo "👉 tmux ls   (cek semua session)"
