# Documentation Policy

Documentation should:

- Reflect the implementation (not aspirational)
- Remain concise and focused
- Avoid redundancy across files
- Be updated only when relevant to the requested change
- Be stored in `docs_src/` for solid, tracked findings
- Be written for future reference, not just immediate use

## What Not to Document

- Do not generate documentation solely for the sake of documentation
- Skip explanatory prose when code is self-documenting
- Avoid duplicating information across multiple files
- Do not over-document transient work in `working_notes/`

## Where to Document

| Location | Purpose | Tracked |
|----------|---------|---------|
| `docs_src/` | Methodology, findings, system design | Yes |
| `working_notes/` | Exploration notes, reference materials | No |
| Code comments | Non-obvious WHY, hidden constraints, workarounds | N/A |
| Commit messages | Context for changes | Yes |

## Diagrams and Visualizations

- **Always use Mermaid diagrams** for data pipelines, process flows, and architecture — not ASCII art
- Mermaid syntax: ```` ```mermaid ... ``` ````
- Mermaid renders natively in mkdocs and supports flowcharts, graphs, sequence diagrams, and more
- ASCII art becomes unreadable when edited, hard to version, and can't be styled
