# Task Plan: Implementering af ClawRAG-forbedringer i Buddy-RAG

## Goal
Opnå et stabilt, effektivt og funktionelt RAG-system, der overgår ClawRAG's kernetiltag for ingestion og danner grundlag for fremtidige udvidelser (Hybrid Search, Robusthed).

## Current Phase
Phase 2: Systemtest & Validering

## Phases

### Phase 1: Ingestion Pipeline Robusthed & Ydeevne (ClawRAG-rettelser)
- [x] **Trin 1: Refactor `ingestion.py` til korrekt `IngestionPipeline` brug.**
  - Send `LlamaDocument` objekter til pipelinen.
  - Lad pipelinen håndtere `SentenceSplitter` og `Settings.embed_model` som transformationer.
  - **Status:** complete

- [x] **Trin 2: Implementer Metadata Sanitering i `ingestion.py`.**
  - Konverter komplekse metadata til simple typer eller JSON-strenge, før de sendes til Qdrant.
  - **Status:** complete

- [x] **Trin 3: Bekræft optimering af Embedding Model.**
  - Fasthold brugen af `nomic-embed-text` for Ollama. Verificer hastighed.
  - **Status:** complete

- [x] **Trin 4: Verificer forbedret Progress Feedback.**
  - Sørg for at `show_progress=True` virker korrekt i `pipeline.run()`.
  - **Status:** complete

### Phase 2: Systemtest & Validering
- [x] Kør en komplet ingestion test med X100-manualen.
  - **Status:** partial (Kørt med X100-Quick-Start-Guide succesfuldt. OperationGuide kører pt.)
- [x] Verificer at Qdrant collection oprettes korrekt og indeholder data.
  - **Status:** complete (Verificeret med `test_retrieval.py` og `curl`).
- [ ] Test et simpelt query mod det nye index.
  - **Status:** in_progress (Venter på LLM integration eller retrieval-succes).

### Phase 3: Fejlhåndtering & Robusthed (Indledende)
- [ ] Evaluer behov for Circuit Breakers og Fallback Chains i `ingestion.py` (ClawRAG-inspireret).
- [ ] Implementer grundlæggende fejlhåndtering for Qdrant-forbindelse.
- **Status:** pending

### Phase 4: Hybrid Search (BM25) Integration
- [ ] Undersøg ClawRAG's `bm25_index.py` og implementer BM25-baseret hybrid search.
- [ ] Integrer Reciprocal Rank Fusion (RRF) for at kombinere søgeresultater.
- **Status:** pending

### Phase 5: Kontinuerlig Forbedring & Udvidelser
- [ ] Overvej Parent-Child Chunking.
- [ ] Overvej Modulær Arkitektur Refactoring.
- [ ] Regelmæssig review af ClawRAG for nye opdateringer.
- **Status:** pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| `planning-with-files` skill | Ideel til filbaseret planlægning, gennemsigtig kode, struktureret tilgang. |
| `audit-code` skill | Vigtig for kode-sikkerhed, risiko vurderet som acceptabel efter kode-gennemgang. |
| Dropped `bobagent-n8n` & `4to1-planner` | Manglende kildekode-transparens og sikkerhedsadvarsler gjorde installation uansvarlig. |
| Switch to `nomic-embed-text` | Prioriterer hastighed og robusthed som anbefalet i RAG-rapporten. |
| `sanitize_metadata` implementeret | Forhindrer Qdrant-fejl ved komplekse metadata-strukturer. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| `init-session.sh` Perm. Denied | `chmod +x` |
| `init-session.sh` File Not Found | Konvertering af CRLF til LF linjeafslutninger. |
| `pip3` not found | Installeret `python3-pip` via `apt-get`. |
