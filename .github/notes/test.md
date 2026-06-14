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