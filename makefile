all:
	python ./src/main.py

clean:
	rm -f images/*.*
	rm -rf ./src/__pycache__/

.PHONY: all clean