# Makefile commands

The project uses a `Makefile` to wrap common development tasks inside Docker containers.

## Commands

| Target | Description |
|--------|-------------|
| `make test` | Runs the test suite in a Docker container with `pytest`. |
| `make format` | Auto-formats the codebase using `ruff`. |
| `make lint` | Lints the codebase using `ruff check`. |

## Containers used

- **test**: `hmcvlab/computer-vision:latest`
- **format / lint**: `ghcr.io/astral-sh/ruff:0.15.17`

All commands mount the current directory as `/app` inside the container.
