# Coding Standards

## No Emojis

All code and documentation must be text-based and professional:
- Source code files
- Documentation and markdown files
- Comments
- Commit messages
- Variable names or filenames

## Code Style

- Follow Python PEP 8 conventions
- Use meaningful variable and function names
- Minimal comments (only WHY, not WHAT)
- Keep functions focused and single-purpose

## Data Operations

- Use **polars** exclusively (not pandas)
- Leverage lazy evaluation and columnar storage
- Chain methods for clarity: `.filter()`, `.select()`, `.with_columns()`
- All CSV/parquet I/O through polars
