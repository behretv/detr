.PHONY: test
test:
	docker run --tty --rm \
		-v .:/app \
		-w /app \
		--user ubuntu \
		hmcvlab/detr:latest \
		bash -c "python -m pytest test/"

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

.PHONY: docker
docker:
	@echo "FROM hmcvlab/computer-vision:latest" > Dockerfile
	@echo "COPY . ." >> Dockerfile
	@echo "RUN pip install -e ." >> Dockerfile
	docker build -t hmcvlab/detr:latest .
	rm Dockerfile

.PHONY: install-hooks
install-hooks:
	@echo "make format && make lint" > .git/hooks/pre-commit
	@echo "make test" > .git/hooks/pre-push
	@chmod +x .git/hooks/pre-commit .git/hooks/pre-push