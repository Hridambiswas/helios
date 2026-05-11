# Contributing to Helios

Thank you for your interest in contributing!

## Branching Strategy

```
main        ← production releases (merge via PR from develop only)
develop     ← integration branch (always deployable)
  ├── feature/*    ← new features (branch from develop)
  ├── fix/*        ← bug fixes (branch from develop)
  ├── hotfix/*     ← critical fixes (branch from main, merge to main + develop)
  ├── refactor/*   ← refactoring (branch from develop)
  └── docs/*       ← documentation only (branch from develop)
```

## Workflow

```bash
# Start a feature
git checkout develop
git pull origin develop
git checkout -b feature/your-feature

# Work, commit, push
git commit -m "feat: add your feature"
git push origin feature/your-feature

# Open PR against develop (not main)
```

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org):

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code change without feature/fix |
| `test:` | Adding or fixing tests |
| `chore:` | Build, CI, dependencies |
| `perf:` | Performance improvement |

## Code Standards

- Python: follow existing style, type hints required
- TypeScript: strict mode, no `any` unless unavoidable
- No commented-out code
- Tests for new logic in `tests/`

## Pull Request Checklist

- [ ] Branched from `develop`
- [ ] Conventional commit messages
- [ ] Tests added/updated
- [ ] No new linting errors
- [ ] README updated if user-facing change
