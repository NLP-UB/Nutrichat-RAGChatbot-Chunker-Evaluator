#!/bin/bash
DATASET="data/ground-truth/QA-Dataset.csv"
OUTPUT="./outputs"

METHODS=("semantic" "recursive" "doublepass")

for method in "${METHODS[@]}"; do
  session="eval_$method"
  echo "🚀 Starting tmux session: $session"
  tmux new-session -d -s "$session" \
    "python evaluate_method.py $method $DATASET $OUTPUT false; read"
done

echo "✅ All experiments launched in tmux sessions."
echo "👉 Use 'tmux ls' to see sessions."
