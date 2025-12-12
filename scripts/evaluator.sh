#!/bin/bash
QnA="data/ground-truth/QA-Dataset.csv"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT="./outputs/$TIMESTAMP"

METHODS=("semantic" "recursive" "doublepass")
EMBEDDERS=("embeddinggemma" "all-minilm" "qwen3-embedding")

declare -A FORMATS=(
  [0]="1 paragraf ringkas padat jelas"
  [1]="poin utama dan penjelasan singkat"
  [2]="jawaban dalam bentuk tabel teks"
)

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

# 1. Start qdrant in its own tmux session
QDRANT_SESSION="qdrant"
if tmux has-session -t $QDRANT_SESSION 2>/dev/null; then
  echo "⚠️ Qdrant session already running."
else
  echo "🚀 Starting Qdrant server in tmux session: $QDRANT_SESSION"
  tmux new-session -d -s "$QDRANT_SESSION" "./qdrant; read"
  # Wait for qdrant to be ready
  sleep 5
fi

# ---------------------------------------------
# 2. NORMAL 3×3×3 NESTED LOOPS (27 processes)
#    but each method uses its OWN port
# ---------------------------------------------
echo "🚀 Launching evaluation processes..."

for method in "${METHODS[@]}"; do
  port=${METHOD_PORT[$method]}
  ollama_url="http://127.0.0.1:$port"

  for embedder in "${EMBEDDERS[@]}"; do
    for format_index in "${!FORMATS[@]}"; do

      format="${FORMATS[$format_index]}"
      session="eval-$method-$embedder-$format_index"

      echo "  🚀 Starting $session using $ollama_url"

      tmux new-session -d -s "$session" \
        "python evaluate_method.py \
          $QnA \
          $method \
          $embedder \
          $format_index \
          \"$format\" \
          $OUTPUT \
          $ollama_url \
          true; read"
    done
  done
done

echo "✅ All experiments launched."
echo "👉 Use 'tmux ls' to see sessions."
echo "👉 Use 'tmux attach -t ollama' to view Ollama logs"