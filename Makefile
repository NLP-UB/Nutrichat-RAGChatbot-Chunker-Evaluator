SCRIPTS := scripts

.PHONY: evaluate kill check-semantic check-doublepass check-recursive test

kill:
	bash $(SCRIPTS)/kill-eval.sh

evaluate:
	bash $(SCRIPTS)/evaluator.sh

check-semantic:
	tmux attach -t eval-semantic

check-recursive:
	tmux attach -t eval-recursive
	
check-doublepass:
	tmux attach -t eval-doublepass

test:
	bash $(SCRIPTS)/whichisit.sh