TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT="./outputs/$TIMESTAMP"

METHODS=("semantic" "recursive" "doublepass")
EMBEDDERS=("embeddinggemma" "all-minilm" "qwen3-embedding")

# 1. Start Ollama in its own tmux session (first)
OLLAMA_SESSION="ollama"
if tmux has-session -t $OLLAMA_SESSION 2>/dev/null; then
  echo "⚠️ Ollama session already running."
else
  echo "🚀 Starting Ollama server in tmux session: $OLLAMA_SESSION"
  tmux new-session -d -s "$OLLAMA_SESSION" "~/ollama/bin/ollama serve; read"
  # Wait for Ollama to start
  sleep 10
fi

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

for method in "${METHODS[@]}"; do
  for embedder in "${EMBEDDERS[@]}"; do
    session="index-$method-$embedder"
    echo "🚀 Starting tmux session: $session"
    tmux new-session -d -s "$session" \
      "python index_data.py $method $embedder; read"
  done
done

echo "✅ All experiments launched."
echo "👉 Use 'tmux ls' to see sessions."
echo "👉 Use 'tmux attach -t ollama' to view Ollama logs"