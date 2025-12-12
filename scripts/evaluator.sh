#!/bin/bash
QnA="data/ground-truth/QA-Dataset.csv"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT="./outputs/$TIMESTAMP"

mkdir -p "$OUTPUT"

METHODS=("semantic" "recursive" "doublepass")
EMBEDDERS=("embeddinggemma" "all-minilm" "qwen3-embedding")

# Map each method → a specific Ollama port
declare -A METHOD_PORT=(
  ["semantic"]=11435
  ["recursive"]=11436
  ["doublepass"]=11437
)

# ----------------------------
# 1. Start 3 Ollama Servers
# ----------------------------
echo "🚀 Starting 3 Ollama servers..."

for method in "${METHODS[@]}"; do
    port=${METHOD_PORT[$method]}
    session="ollama-$method"

    if tmux has-session -t "$session" 2>/dev/null; then
        echo "⚠️ $session already running."
    else
        echo "🚀 Starting $session on port $port"
        tmux new-session -d -s "$session" \
          "export OLLAMA_HOST=127.0.0.1:$port; ~/ollama/bin/ollama serve; read"
        sleep 5
    fi
done

# ----------------------------
# Start Qdrant
# ----------------------------
QDRANT_SESSION="qdrant"
if tmux has-session -t $QDRANT_SESSION 2>/dev/null; then
  echo "⚠️ Qdrant session already running."
else
  echo "🚀 Starting Qdrant server"
  tmux new-session -d -s "$QDRANT_SESSION" "./qdrant; read"
  sleep 5
fi

# ---------------------------------------------
# 2. 3×3 LOOP (method × embedder)
# ---------------------------------------------
echo "🚀 Launching evaluation processes..."

for method in "${METHODS[@]}"; do
  port=${METHOD_PORT[$method]}
  ollama_url="http://127.0.0.1:$port"

  for embedder in "${EMBEDDERS[@]}"; do

    session="eval-$method-$embedder"

    echo "  🚀 Starting $session using $ollama_url"

    tmux new-session -d -s "$session" \
      "python evaluate_method.py \
        $QnA \
        $method \
        $embedder \
        $OUTPUT \
        $ollama_url \
        true; read"
  done
done

echo "✅ All experiments launched."
echo "👉 Use 'tmux ls' to see sessions."
echo "👉 Use 'tmux attach -t ollama' to view Ollama logs"
