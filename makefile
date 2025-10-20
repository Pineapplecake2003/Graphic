ifeq ($(or $(IN_NIX_SHELL),$(NIX_PATH)),)

PYTHON = python
else
ifneq ($(shell command -v nixGLIntel 2>/dev/null),)
PYTHON = nixGLIntel python
else
PYTHON = python
endif
endif

SRC = ./src/main.py
IMAGES = ./images/*.*

all:
	$(PYTHON) $(SRC)

clean:
	rm -f $(IMAGES)
	rm -rf ./src/__pycache__/

.PHONY: all clean