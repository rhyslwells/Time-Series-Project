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

- Use **polars** exclusively (not pandas) for all CSV/parquet I/O and dataframe work
- See [DATA_STACK.md](DATA_STACK.md) for the polars/numpy boundary and API details
