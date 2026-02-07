# Task Completion Checklist

When a coding task is completed, follow these steps to ensure quality:

## 1. Code Quality
- [ ] Format code with `black .`
- [ ] Fix linting issues with `ruff check . --fix`
- [ ] Run type checking with `mypy src/`
- [ ] Ensure all type hints are present for public APIs

## 2. Testing
- [ ] Write or update tests for new functionality
- [ ] Run `pytest` to ensure all tests pass
- [ ] Verify test coverage remains above 85%
- [ ] Run specific test categories if needed:
  - `pytest -m unit` for unit tests
  - `pytest -m integration` for integration tests

## 3. Validation
- [ ] Run `make validate` for full validation (lint + type + test + structure)
- [ ] Verify the fix/feature works as expected manually
- [ ] Check for regressions in related functionality

## 4. Documentation
- [ ] Update module-level CLAUDE.md if structural changes were made
- [ ] Update relevant docstrings
- [ ] Add or update examples if needed
- [ ] Update README.md if user-facing changes were made

## 5. Git Workflow
- [ ] Stage changes with `git add`
- [ ] Commit with descriptive message following conventional commits
- [ ] Push changes when ready
- [ ] Create pull request if working on a branch

## Code Review Preparation
Before requesting review:
- [ ] Ensure all tests pass locally
- [ ] Check that `make lint` and `make type` succeed
- [ ] Verify the changes solve the intended problem
- [ ] Add comments explaining complex logic if necessary

## Special Cases

### For Bug Fixes
- [ ] Add regression test
- [ ] Verify fix doesn't break existing functionality
- [ ] Document the bug and fix if appropriate

### For New Features
- [ ] Add comprehensive tests
- [ ] Update documentation
- [ ] Consider backward compatibility
- [ ] Add feature flags if needed

### For Refactoring
- [ ] Ensure behavior remains unchanged
- [ ] Update tests if needed (not behavior, just structure)
- [ ] Improve code readability
- [ ] Add type hints if missing

### For Performance Improvements
- [ ] Run benchmarks before and after
- [ ] Document performance gains
- [ ] Ensure no regressions in functionality
