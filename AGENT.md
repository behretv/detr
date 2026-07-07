# Goal of this project

A model library that is built up on modern torchvision and utilizes this library heavily.

Using DETR models should be compatible with torchvision models for object detection and instance segmentation.
In particular, models should be trained and saved that can be loaded with torch and has the same behavior
(forward function in eval() and train() mode) ad torchvision models.

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

# Test best practice

## DO's

To reduce the lines of code ind improve maintainability use pytest functionality e.g. `pytest.mark.parametrize`

Try to tests only one thing at a time (if reasonable) and keep the tests short.

Organize test the following way:
 1. Arrange - prepare data for the tests, sometimes there is only act then you can skip the arrange section
 2. Act - execute the function / class you want to test (include the construction of the object if the constructor is part of the test)
 3. Assert - check results

here is an example (also include the comments into every test):

```python
def test_function():
    # Arrange
    a = [1, 2, 3]

    # Act
    result = function(a)

    # Assert
    assert result == 3
```

 ## Dont's

 Keep tests simple, do NOT use complicated if-else statements inside tests
