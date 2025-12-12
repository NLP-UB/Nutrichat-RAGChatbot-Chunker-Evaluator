SCRIPTS := scripts

.PHONY: idx eval kill semantic doublepass recursive test qdrant ollama count

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

ollama:
	tmux attach -t ollama

0-0-0:
	tmux attach -t eval-semantic-embeddinggemma-0
0-0-1:
	tmux attach -t eval-semantic-embeddinggemma-1
0-0-2:
	tmux attach -t eval-semantic-embeddinggemma-2
0-1-0:
	tmux attach -t eval-semantic-all-minilm-0
0-1-1:
	tmux attach -t eval-semantic-all-minilm-1
0-1-2:
	tmux attach -t eval-semantic-all-minilm-2
0-2-0:
	tmux attach -t eval-semantic-qwen3-embedding-0
0-2-1:
	tmux attach -t eval-semantic-qwen3-embedding-1
0-2-2:
	tmux attach -t eval-semantic-qwen3-embedding-2
1-0-0:
	tmux attach -t eval-recursive-embeddinggemma-0
1-0-1:
	tmux attach -t eval-recursive-embeddinggemma-1
1-0-2:
	tmux attach -t eval-recursive-embeddinggemma-2
1-1-0:
	tmux attach -t eval-recursive-all-minilm-0
1-1-1:
	tmux attach -t eval-recursive-all-minilm-1
1-1-2:
	tmux attach -t eval-recursive-all-minilm-2
1-2-0:
	tmux attach -t eval-recursive-qwen3-embedding-0
1-2-1:
	tmux attach -t eval-recursive-qwen3-embedding-1
1-2-2:
	tmux attach -t eval-recursive-qwen3-embedding-2
2-0-0:
	tmux attach -t eval-doublepass-embeddinggemma-0
2-0-1:
	tmux attach -t eval-doublepass-embeddinggemma-1
2-0-2:
	tmux attach -t eval-doublepass-embeddinggemma-2
2-1-0:
	tmux attach -t eval-doublepass-all-minilm-0
2-1-1:
	tmux attach -t eval-doublepass-all-minilm-1
2-1-2:
	tmux attach -t eval-doublepass-all-minilm-2
2-2-0:
	tmux attach -t eval-doublepass-qwen3-embedding-0
2-2-1:
	tmux attach -t eval-doublepass-qwen3-embedding-1
2-2-2:
	tmux attach -t eval-doublepass-qwen3-embedding-2

count:
	ls -l outputs/2025-12-12_14-04-57 | wc -l