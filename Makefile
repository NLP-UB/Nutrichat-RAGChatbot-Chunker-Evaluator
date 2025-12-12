SCRIPTS := scripts

.PHONY: idx eval kill semantic doublepass recursive test qdrant ollama count users

kill:
	bash $(SCRIPTS)/kill-eval.sh

idx:
	bash $(SCRIPTS)/indexer.sh

eval:
	bash $(SCRIPTS)/evaluator.sh
leval:
	bash $(SCRIPTS)/lite-evaluator.sh

semantic:
	tmux attach -t eval-semantic

recursive:
	tmux attach -t eval-recursive
	
doublepass:
	tmux attach -t eval-doublepass

test:
	bash $(SCRIPTS)/whichisit.sh

qdrant:
	tmux attach -t qdrant

o-0:
	tmux attach -t ollama-semantic
o-1:
	tmux attach -t ollama-recursive
o-2:
	tmux attach -t ollama-doublepass

0-0:
	tmux attach -t eval-semantic-embeddinggemma
0-1:
	tmux attach -t eval-semantic-all-minilm
0-2:
	tmux attach -t eval-semantic-qwen3-embedding
1-0:
	tmux attach -t eval-recursive-embeddinggemma
1-1:
	tmux attach -t eval-recursive-all-minilm
1-2:
	tmux attach -t eval-recursive-qwen3-embedding
2-0:
	tmux attach -t eval-doublepass-embeddinggemma
2-1:
	tmux attach -t eval-doublepass-all-minilm
2-2:
	tmux attach -t eval-doublepass-qwen3-embedding

users:
	nvidia-smi --query-compute-apps=pid --format=csv,noheader | \
	xargs -I{} ps -o user= -p {} | sort | uniq

count:
	ls -l outputs/2025-12-12_14-04-57 | wc -l