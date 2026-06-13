.PHONY: test
test:
	docker run --tty --rm \
		-v .:/app \
		-w /app \
		hmcvlab/computer-vision:latest \
		bash -c "pip install -e . && python -m pytest test/"

.PHONY: format
format:
	docker run --tty --rm \
		-v .:/app \
		-w /app \
		ghcr.io/astral-sh/ruff:0.15.17 \
		format

.PHONY: lint
lint:
	docker run --tty --rm \
		-v .:/app \
		-w /app \
		ghcr.io/astral-sh/ruff:0.15.17 \
		check

