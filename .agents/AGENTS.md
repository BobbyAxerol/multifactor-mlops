# Agent Behavioral Rules & Constraints

## Git & Branch Management Rules
- **NEVER Automatically Merge or Push to `main`**:
  - The agent MUST NOT automatically execute `git merge` into the `main` branch or run `git push origin main` unless the user explicitly instructs or approves it.
  - All feature development, experimental changes, and commits must remain strictly on the active working branch (e.g., `feat/multifactor-macro-features` or `dev`).
  - Merging to `main` and pushing to `origin/main` requires explicit user confirmation.
