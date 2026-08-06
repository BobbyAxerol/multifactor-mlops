# Contributing to Multifactor Portfolio

Thank you for your interest in contributing to the **Multifactor Portfolio** project!

---

## 🚀 How to Contribute

### 1. Reporting Bugs
* Search existing issues to ensure the bug hasn't been reported.
* Open a new bug report issue using the [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).
* Include detailed steps to reproduce the issue, environment info, and log tracebacks.

### 2. Suggesting Enhancements
* Open a feature request issue using the [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md).
* Explain the use case, why this feature is valuable, and how it aligns with the two-layered quantitative pipeline.

### 3. Submitting Pull Requests
1. Fork or clone the repository.
2. Create a topic branch following our branch naming guidelines (`feat/<feature-name>`, `fix/<bug-name>`, or `research/<experiment-name>`).
3. Write modular, documented Python code adhering to PEP 8 standards.
4. Ensure all unit tests pass:
   ```bash
   poetry run pytest tests/ -q
   ```
5. Format code with `black`:
   ```bash
   poetry run black src/multifactor_mlops/ tests/
   ```
6. Submit a Pull Request following the [PR Template](.github/PULL_REQUEST_TEMPLATE.md).

---

## 🌿 Branch Naming Guidelines

* `feat/`: New strategy features, factor models, or data sources.
* `fix/`: Bug fixes or code correction.
* `research/`: Research experiments, feature analysis, or alternative backtest backends.
* `chore/`: Dependency updates, CI/CD, or documentation.

---

## 🧪 Testing Guidelines

* Every new factor calculation or data pipeline modification must include unit tests in `tests/`.
* Tests must pass cleanly with zero errors before merging.
