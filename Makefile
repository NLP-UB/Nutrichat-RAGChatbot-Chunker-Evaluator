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

e0-0:
	tmux attach -t eval-semantic-embeddinggemma
e0-1:
	tmux attach -t eval-semantic-all-minilm
e0-2:
	tmux attach -t eval-semantic-qwen3-embedding
e1-0:
	tmux attach -t eval-recursive-embeddinggemma
e1-1:
	tmux attach -t eval-recursive-all-minilm
e1-2:
	tmux attach -t eval-recursive-qwen3-embedding
e2-0:
	tmux attach -t eval-doublepass-embeddinggemma
e2-1:
	tmux attach -t eval-doublepass-all-minilm
e2-2:
	tmux attach -t eval-doublepass-qwen3-embedding
	
i0-0:
	tmux attach -t index-semantic-embeddinggemma
i0-1:
	tmux attach -t index-semantic-all-minilm
i0-2:
	tmux attach -t index-semantic-qwen3-embedding
i1-0:
	tmux attach -t index-recursive-embeddinggemma
i1-1:
	tmux attach -t index-recursive-all-minilm
i1-2:
	tmux attach -t index-recursive-qwen3-embedding
i2-0:
	tmux attach -t index-doublepass-embeddinggemma
i2-1:
	tmux attach -t index-doublepass-all-minilm
i2-2:
	tmux attach -t index-doublepass-qwen3-embedding

users:
	nvidia-smi --query-compute-apps=pid --format=csv,noheader | \
	xargs -I{} ps -o user= -p {} | sort | uniq

count:
	ls -l outputs/2025-12-12_14-04-57 | wc -l

ollama:
	tmux attach -t ollama