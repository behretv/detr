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


.PHONY: install-hooks
install-hooks:
	@echo "make format && make lint" > .git/hooks/pre-commit
	@echo "make test" > .git/hooks/pre-push
	@chmod +x .git/hooks/pre-commit .git/hooks/pre-push