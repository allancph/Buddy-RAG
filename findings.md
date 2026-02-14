# Findings & Decisions

## Requirements
-

## Research Findings
-

## Technical Decisions
| Decision | Rationale |
|----------|-----------|

## Issues Encountered
| Issue | Resolution |
| `chunk_size` for `nomic-embed-text` | Research confirms a practical limitation of **512 tokens** for the `nomic-embed-text` model in some implementations, despite a theoretical max of 8192. Our own error (`the input length exceeds the context length`) validates this practical limit. The `chunk_size` in `ingestion.py` must be set below this threshold. |
|-------|------------|

## Resources
-
