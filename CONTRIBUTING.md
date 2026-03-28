# Contributing to LSTM Amazon Stock Price Predictor

Thank you for considering contributing! This document explains the process for reporting bugs, requesting features, and submitting pull requests.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Report a Bug](#how-to-report-a-bug)
3. [How to Request a Feature](#how-to-request-a-feature)
4. [Development Setup](#development-setup)
5. [Branching Strategy](#branching-strategy)
6. [Submitting a Pull Request](#submitting-a-pull-request)
7. [Code Style](#code-style)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold it.

---

## How to Report a Bug

1. Search [existing issues](../../issues) to avoid duplicates.
2. Open a new issue and include:
   - A clear, descriptive title
   - Steps to reproduce
   - Expected vs. actual behaviour
   - Your environment (OS, Python version, PyTorch version, CUDA version)
   - Any relevant error messages or stack traces

---

## How to Request a Feature

1. Open an issue with the label **enhancement**.
2. Describe the problem your feature would solve and your proposed solution.
3. If you plan to implement it yourself, mention that in the issue so we can coordinate.

---

## Development Setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/LSTM_air_prediction.git
cd LSTM_air_prediction

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Branching Strategy

| Branch type     | Naming convention          | Purpose                          |
|-----------------|----------------------------|----------------------------------|
| Feature         | `feature/<short-name>`     | New functionality                |
| Bug fix         | `fix/<short-name>`         | Fix for a reported bug           |
| Documentation   | `docs/<short-name>`        | Documentation-only changes       |
| Experiment      | `exp/<short-name>`         | Model experiments / explorations |

Always branch off `main`:

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

---

## Submitting a Pull Request

1. Make sure your branch is up to date with `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. Run the scripts to verify nothing is broken:
   ```bash
   python src/LSTM_training.py
   ```
3. Push your branch and open a pull request against `main`.
4. Fill in the PR template:
   - What does this PR change?
   - How was it tested?
   - Any related issues? (use `Closes #<issue-number>`)

PRs are reviewed within a reasonable time. At least one approval is required before merging.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use type hints for all function signatures.
- Keep functions focused and small.
- Prefer descriptive variable names over single letters (except conventional loop indices).
- Do not commit Jupyter notebook output cells — clear outputs before committing:
  ```bash
  jupyter nbconvert --clear-output --inplace src/notebooks/*.ipynb
  ```
