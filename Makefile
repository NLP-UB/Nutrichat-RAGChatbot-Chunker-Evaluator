SCRIPTS := scripts

.PHONY: idx eval kill semantic doublepass recursive test qdrant ollama

kill:
	bash $(SCRIPTS)/kill-eval.sh

idx:
	bash $(SCRIPTS)/indexer.sh

eval:
	bash $(SCRIPTS)/evaluator.sh

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