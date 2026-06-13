# Test best practice

## DO's

To reduce the lines of code ind improve maintainability use pytest functionality e.g. `pytest.mark.parametrize`

Organize test the following way:
 1. Arrange - prepare data for the tests
 2. Act - execute function you want to test
 3. Assert - check results

 ## Dont's

 Keep tests simple, do NOT use complicated if-else statements inside tests