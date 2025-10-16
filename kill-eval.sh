#!/bin/bash
# Kill only eval_* and qdrant sessions
for session in $(tmux ls | grep -E 'eval_|qdrant' | cut -d: -f1); do
  echo "Killing $session ..."
  tmux kill-session -t "$session"
done
