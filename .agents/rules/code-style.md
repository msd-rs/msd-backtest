---
trigger: always_on
---

- Language: `Python`
    - Indentation: 2 spaces
- Package manager: `uv`
    - Use `uv add` to add dependencies
    - Use `uv run` to run commands
- Testing Framework: `pytest`
    - Add test in 'tests' folder
    - Unit test function should have 'test_' prefix
    - Use `uv run pytest` to run tests