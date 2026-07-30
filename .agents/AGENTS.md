# Agent Behavioral Rules & Constraints

## Git & Branch Management Rules
- **NEVER Automatically Merge or Push to `main`**:
  - The agent MUST NOT automatically execute `git merge` into the `main` branch or run `git push origin main` unless the user explicitly instructs or approves it.
  - All feature development, experimental changes, and commits must remain strictly on the active working branch (e.g., `feat/multifactor-macro-features` or `dev`).
  - Merging to `main` and pushing to `origin/main` requires explicit user confirmation.

# Rules

- Mỗi lần chỉnh sửa code xong, bắt buộc phải commit các thay đổi lên nhánh `dev`. Tuyệt đối không checkout qua nhánh `main` để commit. Nhánh `main` được bảo vệ bằng pre-commit hook chặn mọi commit trực tiếp.

- Luôn không được tự ý train hay optuna tunning với trials lớn nếu không được phép. ĐƯợc phép nhỏ để debug syntax thôi.

- Sủa mã nguồn lớn, thay đổi ý tưởng bất kỳ chỗ nào thì phải luôn hỏi lại xem tôi có đồng ý không rồi mới implement.

- After each completed code/config/documentation fix, create a git commit with a clear message in the affected repository, unless the user explicitly asks not to commit.
- Before committing, check the repository status and avoid staging unrelated user work from other repositories.
- Before committing, verify `git config user.name` and `git config user.email`; never commit with a system/default identity. Use the user's configured git identity for that repository.
- Commit messages should briefly name the subsystem and the concrete behavior fixed or added.
- Always commit every change to a feature/dev branch, and never commit directly to the `main` branch.
- Never implement major refactors or change the implementation direction without obtaining the user's explicit approval first.
- After every major code change, always review the entire integration flow to ensure all components are fully adapted, checking for logic errors and synchronization across files.



