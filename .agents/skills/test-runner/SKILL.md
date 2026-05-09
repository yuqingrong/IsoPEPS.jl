# test-runner

Run Julia tests with coverage reporting and detailed output.

## Usage
```
/test-runner [options]
```

## Options
- No arguments: Run all tests
- `--coverage`: Generate coverage report
- `--verbose`: Show detailed test output
- `--file <path>`: Run specific test file

## Workflow

1. **Check test environment**
   - Verify Project.toml and test dependencies
   - Ensure test directory exists

2. **Run tests**
   - Execute `julia --project=. -e 'using Pkg; Pkg.test()'`
   - Capture output and parse results
   - Report test summary (passed/failed/errored)

3. **Coverage analysis** (if requested)
   - Generate coverage data
   - Report coverage percentage
   - Identify untested code paths

4. **Handle failures**
   - Display failed test details
   - Show stack traces
   - Suggest potential fixes if patterns are recognized

## Success Criteria
- All tests pass
- Coverage meets threshold (if specified)
- No deprecation warnings

## Example Output
```
Running tests for IsoPEPS.jl...
✓ Gates tests (15 passed)
✓ Observables tests (23 passed)
✓ Training tests (12 passed)
✓ Transfer matrix tests (18 passed)
✓ Visualization tests (8 passed)

Total: 76 tests passed, 0 failed
Coverage: 87.3%
```
