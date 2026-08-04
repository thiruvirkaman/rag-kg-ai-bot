# rag-kg-ai-bot

A chat application that answers questions about **Tamil Nadu government farmer welfare schemes** using a **Knowledge-Graph RAG** (Retrieval-Augmented Generation) architecture.

Data is scraped one-time from the [TN Agriculture - Farmers Welfare Department schemes listing](https://www.tn.gov.in/scheme_list.php?dep_id=Mg==), then entities & relationships are extracted, stored in a **Neo4j** graph database, and queried via **GraphRAG hybrid retrieval** to ground LLM answers.

---

## Quick Start (5 commands)

```powershell
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start Neo4j graph database (Docker)
docker compose up -d

# 3. Create .env from template and add your API keys
Copy-Item .env.example .env
#    -> edit .env: set LLM_API_KEY, LLM_BASE_URL, model names

# 4. Build the knowledge graph (scrape -> extract -> load into Neo4j)
python -m app.ingest

# 5. Launch the chat UI
streamlit run app/streamlit_app.py
```

That's it. Open the Streamlit URL (usually http://localhost:8501) and ask questions about farmer schemes.

---

## Detailed Setup

### Step 1 — Install Python dependencies
```powershell
pip install -r requirements.txt
```
Requires **Python 3.11+**. Key packages: `neo4j`, `openai`, `streamlit`, `beautifulsoup4`, `pydantic-settings`, `tenacity`, `rich`.

### Step 2 — Start Neo4j
```powershell
docker compose up -d
```
- Bolt: `bolt://localhost:7687`
- Browser: http://localhost:7474
- Credentials: `neo4j` / `password` (change in `docker-compose.yml` if needed)
- Includes APOC plugin and is configured for vector indexes (Neo4j 5.20)

Verify it's running:
```powershell
docker compose ps
```

### Step 3 — Configure environment
```powershell
Copy-Item .env.example .env
```
Edit `.env` with your settings. See **[Configuration](#configuration)** below for details.

#### Ollama Cloud (default setup)
This project uses **Ollama Cloud** for the LLM and a **local Ollama** daemon for embeddings:

```env
# LLM — Ollama Cloud (OpenAI-compatible endpoint)
LLM_BASE_URL=https://ollama.com/v1
LLM_API_KEY=your_ollama_api_key_here        # get from https://ollama.com/settings/keys
LLM_CHAT_MODEL=gemma4:31b                   # no "-cloud" suffix in API calls
LLM_EXTRACT_MODEL=gemma4:31b

# Embeddings — local Ollama (nomic-embed-text, 768-dim)
EMBED_BASE_URL=http://localhost:11434/v1
EMBED_API_KEY=ollama
EMBED_MODEL=nomic-embed-text
EMBED_DIMENSIONS=768
```

> **Note:** Ollama Cloud does not offer cloud embedding models. `nomic-embed-text` runs locally via the Ollama daemon (`ollama serve`). It only runs during ingest (~160 chunks) and once per user question — minimal local load.

Install Ollama and pull the embedding model:
```powershell
ollama pull nomic-embed-text
```

#### Alternative: OpenAI
```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_CHAT_MODEL=gpt-4o-mini
LLM_EXTRACT_MODEL=gpt-4o-mini
EMBED_BASE_URL=https://api.openai.com/v1
EMBED_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small
EMBED_DIMENSIONS=1536
```

### Step 4 — Build the knowledge graph (ingest)
```powershell
python -m app.ingest
```
This runs the full pipeline:

| Step | Module | Output | What it does |
|------|--------|--------|--------------|
| 1. Data Load | `app/scrape.py` | `data/schemes_raw.json` | Scrapes all 54 scheme pages from tn.gov.in |
| 2. Entity Extraction | `app/extract.py` | `data/schemes_chunks.json` | Extracts crops, inputs, districts, eligibility, etc. via rules + LLM |
| 3. Relationship Mapping | `app/extract.py` | `data/schemes_chunks.json` | Maps entities into `(Scheme)-[:REL]->(Entity)` triples |
| 4. Graph Storage | `app/graph_store.py` | Neo4j | Loads nodes, edges, and embedded text chunks into Neo4j |

**Flags:**
- `--force` — re-run every step, ignoring caches (re-scrape + re-extract + reload)
- `--no-llm` — extraction with rules only (no LLM calls; lower quality but free)
- `--skip-load` — skip the Neo4j load (just refresh the JSON caches)

**Run steps individually:**
```powershell
python -m app.scrape        # step 1 only -> data/schemes_raw.json
python -m app.extract       # steps 2+3 only -> data/schemes_chunks.json
python -m app.graph_store   # step 4 only -> Neo4j
```

**Caching:** Steps 1 and 2 cache their output as JSON in `data/`. Re-running `python -m app.ingest` without `--force` loads from cache and only re-runs the Neo4j load. Delete `data/schemes_raw.json` or `data/schemes_chunks.json` to force a refresh of that step.

### Step 5 — Launch the chat UI
```powershell
streamlit run app/streamlit_app.py
```
- Opens at http://localhost:8501
- Ask questions in natural language about farmer schemes
- Answers include citations to source scheme titles + URLs
- Sidebar shows connection status and sample questions

---

## Configuration

All settings are env-driven via `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `https://ollama.com/v1` | OpenAI-compatible LLM endpoint |
| `LLM_API_KEY` | — | API key for LLM provider |
| `LLM_CHAT_MODEL` | `gemma4:31b` | Model for chat answers |
| `LLM_EXTRACT_MODEL` | `gemma4:31b` | Model for entity extraction |
| `EMBED_BASE_URL` | `http://localhost:11434/v1` | Embeddings endpoint |
| `EMBED_API_KEY` | `ollama` | API key for embeddings |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `EMBED_DIMENSIONS` | `768` | Embedding vector dimensions (must match model) |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `RETRIEVAL_TOP_K` | `6` | Number of chunks to retrieve per question |
| `RETRIEVAL_HOPS` | `2` | Graph traversal depth from matched scheme |

> **Important:** `EMBED_DIMENSIONS` must match your embedding model's native output size. `nomic-embed-text` = 768, `text-embedding-3-small` = 1536. The Neo4j vector index is created with this value at load time.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Neo4j load failed: connection refused` | Ensure Docker is running: `docker compose up -d`. Check `docker compose ps`. |
| `LLM extraction failed` errors | Check `LLM_API_KEY` and `LLM_BASE_URL` in `.env`. Test with: `python -c "from app.llm import LLMClient; from app.config import get_settings; print(LLMClient(get_settings()).chat([{'role':'user','content':'say OK'}]))"` |
| Embeddings error / dimension mismatch | Ensure `EMBED_DIMENSIONS` matches your model. Pull the model: `ollama pull nomic-embed-text`. |
| `405 Method Not Allowed` from Ollama | You're using the wrong base URL. Cloud = `https://ollama.com/v1`, local = `http://localhost:11434/v1`. |
| Want to re-scrape from scratch | Delete `data/` folder or run `python -m app.ingest --force` |
| Chat UI can't connect to Neo4j | Verify Neo4j is running and credentials in `.env` match `docker-compose.yml`. |

---

## Architecture

```
tn.gov.in scheme pages
        │
        ▼
[1. Data Load]          app/scrape.py        -> data/schemes_raw.json
        │
        ▼
[2. Entity Extraction]  app/extract.py       -> data/schemes_chunks.json
[3. Relationship Map]     (rule-based + LLM)
        │
        ▼
[4. Graph Storage]      app/graph_store.py   -> Neo4j (nodes, edges, vectors)
        │
        ▼
[5. Graph Query / RAG]  app/rag.py           -> vector search + subgraph traversal
        │
        ▼
[Streamlit Chat UI]     app/streamlit_app.py
```

### Knowledge graph schema

**Nodes**
- `:Scheme` {id, title, url, funding_pattern, description, how_to_avail, ...}
- `:Department`, `:Sponsor` (State / Central / Both)
- `:Beneficiary` (Farmers / Women / SC-ST / ...), `:BenefitType` (Subsidy / Grant / ...)
- `:Crop`, `:Input` (Seeds / Gypsum / Rhizobium / ...), `:Activity`
- `:District`, `:ContactRole`, `:Eligibility`
- `:Chunk` {text, embedding} — embedded text windows, linked to `Scheme`

**Relationships**
`(Scheme)-[:OFFERED_BY]->(Sponsor)`, `RUN_BY`, `TARGETS`, `PROVIDES`,
`SUPPORTS_CROP`, `DISTRIBUTES`, `INCLUDES_ACTIVITY`, `APPLICABLE_IN`,
`APPLY_TO`, `HAS_ELIGIBILITY`, `(Chunk)-[:PART_OF]->(Scheme)`

### Retrieval strategy (GraphRAG hybrid)
1. Embed the question; run a **vector similarity search** over `:Chunk` nodes.
2. From each matched scheme, **traverse the subgraph** to collect neighbour entities (funding, beneficiaries, districts, eligibility, contacts, ...).
3. Assemble a structured context block per scheme (chunk text + graph facts).
4. Feed context + question to the LLM to produce a grounded answer with citations.

## Project structure
```
rag-kg-ai-bot/
├── app/
│   ├── __init__.py
│   ├── config.py          # env-driven settings (pydantic-settings)
│   ├── console.py         # rich logging helpers
│   ├── scrape.py          # [step 1] data load from tn.gov.in
│   ├── extract.py         # [step 2+3] entity extraction + relationship mapping
│   ├── llm.py             # OpenAI-compatible LLM client (retry + JSON parsing)
│   ├── embeddings.py      # OpenAI-compatible embeddings client
│   ├── graph_store.py     # [step 4] Neo4j load (nodes, edges, vector index)
│   ├── rag.py             # [step 5] GraphRAG retrieval + answer synthesis
│   ├── ingest.py          # orchestrates steps 1-4 (CLI)
│   └── streamlit_app.py   # chat UI
├── data/                  # cached scrape + extraction JSON (auto-created)
├── docker-compose.yml     # Neo4j 5.x with APOC
├── requirements.txt
├── pyproject.toml         # ruff config
├── .env.example
└── README.md
```

## Sample questions
- "What schemes are available for farmer training?"
- "Which schemes distribute seeds to farmers?"
- "How do I apply for the Soil Health Card scheme?"
- "What subsidies are available for oilseed farmers?"
- "Which schemes are sponsored by the State government?"
- "What is the funding pattern for Training to Farmers?"

## Data source & scope
- Source: https://www.tn.gov.in/scheme_list.php?dep_id=Mg== (Agriculture - Farmers Welfare Department)
- 54 schemes are scraped one-time and cached to `data/schemes_raw.json`.
- Fields that the source site leaves blank (Income, Age, Community, District, Associated Scheme, Introduced On) are verified-empty on tn.gov.in for this department — not a parser bug.

## Notes
- All LLM JSON output is validated with retries (`tenacity`) and loose JSON extraction (fenced / balanced-brace fallback) for robustness.
- Graph load is idempotent (`MERGE` + uniqueness constraints).
- The vector index (`chunk_embedding`) requires Neo4j 5.13+; the bundled `docker-compose.yml` uses 5.20.