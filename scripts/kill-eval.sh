#!/bin/bash
# Kill only eval-* index-* and qdrant sessions
for session in $(tmux ls | grep -E 'index-|eval-|qdrant|ollama' | cut -d: -f1); do
  echo "Killing $session ..."
  tmux kill-session -t "$session"
done