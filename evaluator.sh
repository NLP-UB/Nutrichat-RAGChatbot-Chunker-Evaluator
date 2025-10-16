#!/bin/bash
DATASET="data/ground-truth/QA-Dataset.csv"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT="./outputs/$TIMESTAMP"

METHODS=("semantic" "recursive" "doublepass")

# 1. Start qdrant in its own tmux session (sequential)
QDRANT_SESSION="qdrant"
if tmux has-session -t $QDRANT_SESSION 2>/dev/null; then
  echo "⚠️ Qdrant session already running."
else
  echo "🚀 Starting Qdrant server in tmux session: $QDRANT_SESSION"
  tmux new-session -d -s "$QDRANT_SESSION" "./qdrant; read"
  # wait a bit for qdrant to be ready
  sleep 5
fi

# 2. Run experiments in parallel tmux sessions
for method in "${METHODS[@]}"; do
  session="eval_$method"
  echo "🚀 Starting tmux session: $session"
  tmux new-session -d -s "$session" \
    "python evaluate_method.py $method $DATASET $OUTPUT false; read"
done

echo "✅ All experiments launched."
echo "👉 Use 'tmux ls' to see sessions."