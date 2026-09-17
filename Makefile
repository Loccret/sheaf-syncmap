PYTHON ?= python
ARGS ?=

.PHONY: ablation_test adaption_test

ablation_test:
	$(PYTHON) -m src.experiments --experiment ablation $(ARGS)

adaption_test:
	$(PYTHON) -m src.experiments --experiment adaption $(ARGS)
