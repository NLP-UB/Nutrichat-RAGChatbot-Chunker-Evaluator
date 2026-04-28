#!/bin/bash
DATASET="data/ground-truth/QA-Dataset.csv"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT="./outputs/$TIMESTAMP"

METHODS=("semantic" "recursive" "doublepass")

OLLAMA_SESSION="ollama"
if tmux has-session -t $OLLAMA_SESSION 2>/dev/null; then
  echo "Ollama session already running."
else
  echo "Starting Ollama server in tmux session: $OLLAMA_SESSION"
  tmux new-session -d -s "$OLLAMA_SESSION" "~/ollama/bin/ollama serve; read"
  sleep 10
fi

QDRANT_SESSION="qdrant"
if tmux has-session -t $QDRANT_SESSION 2>/dev/null; then
  echo "Qdrant session already running."
else
  echo "Starting Qdrant server in tmux session: $QDRANT_SESSION"
  tmux new-session -d -s "$QDRANT_SESSION" "./qdrant; read"
  sleep 5
fi

for method in "${METHODS[@]}"; do
  session="eval-$method"
  echo "Starting tmux session: $session"
  tmux new-session -d -s "$session" \
    "python evaluate_method.py $method $DATASET $OUTPUT false; read"
done

echo "All experiments launched."
echo "Use 'tmux ls' to see sessions."
echo "Use 'tmux attach -t ollama' to view Ollama logs"