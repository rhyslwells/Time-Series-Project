# Response Style

## When Proposing Changes

- Explain reasoning briefly and concretely
- Reference affected components by file and line number
- Highlight architectural implications
- Prefer implementation guidance over abstract discussion
- Keep recommendations aligned with existing repository structure

## Implementation Principles

- **Don't over-engineer**: don't add features, refactoring, or abstractions beyond what the task requires
- **Trust guarantees**: don't add error handling for scenarios that can't happen
- **Validation at boundaries**: only validate external input, not internal state
- **Minimal comments**: only explain WHY (non-obvious constraints, workarounds); never explain WHAT
- **Prefer deletion**: if code is unused, delete it completely (no `_removed` comments)
- **Follow repository patterns**: defer to existing conventions, not generic best practices

## Code Review Standards

- Prioritize correctness and security over style
- Catch OWASP vulnerabilities: command injection, XSS, SQL injection, etc.
- Avoid backwards-compatibility hacks unless essential
- Test feature correctness in the actual UI when applicable

## The Repository First Rule

When repository code, documentation, or architecture differs from general best practices:

1. Follow the repository's established patterns
2. Maintain consistency with existing implementations
3. Recommend alternatives separately if they provide significant benefit
4. Do not automatically rewrite code to match generic best practices

**The repository is the source of truth.**
