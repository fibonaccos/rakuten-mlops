# Some useful colors

RESET	= \033[0m
BOLD	= \033[1m
RED		= \033[31m
GREEN	= \033[32m
YELLOW	= \033[33m
BLUE	= \033[34m
MAGENTA	= \033[35m
CYAN	= \033[36m


# Variables

## for git
BRANCH 	?= 
MESSAGE	?= 

## for uv
GROUP 	?=
PACKAGE	?=


# Make targets
KNOWN_TARGETS := setup add commit push

# Commands

check-branch:
ifndef BRANCH
	printf "$(RED)$(BOLD)ERROR : BRANCH is mandatory.$(RESET)\n"
	printf "$(RED)Usage : make setup BRANCH=<BRANCH_NAME>$(RESET)\n"
endif
	@if git show-ref --verify --quiet refs/heads/$(BRANCH); then \
		printf "$(RED)$(BOLD)Error: branch '$(BRANCH)' already exists in local.$(RESET)\n"; \
		exit 1; \
	elif git ls-remote --heads origin $(BRANCH) | grep -q $(BRANCH); then \
		printf "$(RED)$(BOLD)Error : branch '$(BRANCH)' already exists on remote.$(RESET)\n"; \
		exit 1; \
	fi

check-package:
ifndef PACKAGE
	printf "$(RED)$(BOLD)ERROR : PACKAGE is mandatory.$(RESET)\n"
	printf "$(RED)Usage : make add PACKAGE=<PACKAGE_NAME>$(RESET)\n"
endif

check-precommit:
	@printf "$(MAGENTA)[make] $(BLUE)Checking installation of pre-commit hooks$(RESET)\n"
	@pre-commit install
	@printf "$(GREEN)$(BOLD)Pre-commit hooks installed.$(RESET)\n"


.PHONY: setup add commit push

setup: check-branch
	@printf "$(MAGENTA)[make] $(BLUE)Fetching 'dev' remote branch$(RESET)\n"
	@git fetch origin dev
	@printf "$(MAGENTA)[make] $(BLUE)Creating branch $(BRANCH) based on 'dev'$(RESET)\n"
	@git checkout -b $(BRANCH) origin/dev
	@printf "$(MAGENTA)[make] $(BLUE)Installing dependencies$(RESET)\n"
	@uv sync
	@printf "$(MAGENTA)[make] (GREEN)$(BOLD)Setup completed ! You can start working on your branch.$(RESET)\n"

add: check-package
ifeq ($(GROUP),)
	@printf "$(MAGENTA)[make] $(BLUE)Adding package $(PACKAGE)$(RESET)\n"
	@uv add --group core $(PACKAGE)
else
	@printf "$(MAGENTA)[make] $(BLUE)Adding package $(PACKAGE) to group $(GROUP)$(RESET)\n"
	@uv add --group $(GROUP) $(PACKAGE)
endif
	@printf "$(MAGENTA)[make] $(BLUE)Locking dependencies$(RESET)\n"
	@uv lock
	@printf "$(MAGENTA)[make] $(BLUE)Exporting to requirements$(RESET)\n"
ifeq ($(GROUP),)
	@uv export -qq --group core --no-hashes -o core/requirements.txt
else
	@uv export -qq --group $(GROUP) --no-hashes -o services/$(GROUP)/requirements.txt
endif
	@printf "$(MAGENTA)[make] $(GREEN)$(BOLD)Dependencies successfully added.$(RESET)\n"

commit: check-precommit
	@printf "$(MAGENTA)[make] $(BLUE)Committing changes$(RESET)\n"
	@$(eval TARGETS := $(if $(FILES),$(FILES),$(filter-out $(KNOWN_TARGETS),$(MAKECMDGOALS))))
	@if [ -z "$(TARGETS)" ]; then \
		printf "$(RED)$(BOLD)ERROR : you must specify files to add to the commit.$(RESET)\n"; \
		printf "$(RED)Usage : make commit <FILE1> <FILE2> ... MESSAGE=<MESSAGE>$(RESET)\n" \
		exit 1; \
	fi
	@if [ -z "$(MESSAGE)" ]; then \
		printf "$(RED)$(BOLD)ERROR : MESSAGE is mandatory.$(RESET)\n"; \
		printf "$(RED)Usage : make commit <FILE1> <FILE2> ... MESSAGE=<MESSAGE>$(RESET)\n" \
		exit 1; \
	fi
	@git add $(TARGETS)
	@git commit -m "$(MESSAGE)"
	@printf "$(MAGENTA)[make] $(GREEN)$(BOLD)Files committed to your local branch.$(RESET)\n"
