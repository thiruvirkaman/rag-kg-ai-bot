"""Neo4j graph storage.

Pipeline step 4: Graph Storage.

Loads the extracted scheme graphs into Neo4j using UNWIND + MERGE so the
load is idempotent. Also creates:

  * uniqueness constraints on each node label's `name` (and `Scheme.id`)
  * a vector index over ``Chunk`` nodes for GraphRAG retrieval

Each scheme's free-text (description + how-to-avail + funding) is embedded
and stored as a ``(:Chunk {text, embedding})`` node linked to its scheme via
``(Chunk)-[:PART_OF]->(Scheme)``.
"""
from __future__ import annotations

from typing import Any

from neo4j import Driver, GraphDatabase

from .config import Settings
from .console import get_logger, out
from .embeddings import EmbeddingsClient
from .extract import SchemeGraph, load_cached_graph

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Cypher: schema setup
# --------------------------------------------------------------------------- #

CONSTRAINTS: list[tuple[str, str]] = [
    ("Scheme", "id"),
    ("Department", "name"),
    ("Sponsor", "name"),
    ("Beneficiary", "name"),
    ("BenefitType", "name"),
    ("Crop", "name"),
    ("Input", "name"),
    ("Activity", "name"),
    ("District", "name"),
    ("ContactRole", "name"),
    ("Eligibility", "text"),
    ("Chunk", "chunk_id"),
]


def _ensure_schema(driver: Driver, database: str, embed_dims: int) -> None:
    with driver.session(database=database) as sess:
        for label, prop in CONSTRAINTS:
            if label == "Scheme" and prop == "id":
                cy = (f"CREATE CONSTRAINT {label.lower()}_uniq "
                      f"IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
            else:
                cy = (f"CREATE CONSTRAINT {label.lower()}_uniq "
                      f"IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")
            sess.run(cy)
        # vector index for GraphRAG over Chunk.embedding
        # Neo4j 5.13+ supports `vector_index` via CREATE VECTOR INDEX.
        sess.run(
            """
            CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: $dims,
                `vector.similarity_function`: 'cosine'
              }
            }
            """,
            dims=embed_dims,
        )
    log.info("Schema constraints + vector index ensured.")


# --------------------------------------------------------------------------- #
# Cypher: load scheme + triples
# --------------------------------------------------------------------------- #

# Merge a Scheme node with its scalar properties.
MERGE_SCHEME = """
UNWIND $rows AS row
MERGE (s:Scheme {id: row.scheme_id})
SET s.title = row.title,
    s.url = row.scheme_props.url,
    s.funding_pattern = row.scheme_props.funding_pattern,
    s.introduced_on = row.scheme_props.introduced_on,
    s.scheme_type = row.scheme_props.scheme_type,
    s.associated_scheme = row.scheme_props.associated_scheme,
    s.uploaded_file = row.scheme_props.uploaded_file,
    s.description = row.scheme_props.description,
    s.how_to_avail = row.scheme_props.how_to_avail,
    s.organisation = row.scheme_props.organisation,
    s.department_id = row.scheme_props.department_id
"""

# Merge a generic labelled node + relationship from a Scheme.
# Parameters: rows = [{scheme_id, scheme_title, slabel, sname, rel,
#                      tlabel, tname, tprop}]
MERGE_TRIPLE = """
UNWIND $rows AS row
MATCH (s:Scheme {id: row.scheme_id})
CALL {
  WITH row
  MERGE (t {name: row.tname})
  SET t:`__placeholder__`
  WITH row, t
  REMOVE t:`__placeholder__`
  SET t:row.tlabel
  RETURN t
}
// re-match with the dynamic label so the relationship is typed correctly
CALL {
  WITH row, t
  MATCH (tn)
  WHERE (tn:row.tlabel OR tn.name = row.tname)
    AND elementId(tn) = elementId(t)
  RETURN tn
}
MERGE (s)-[:`__placeholder__`]->(tn)
WITH row, s, tn
// cannot dynamically set rel type in a single MERGE; we use apoc if present,
// else we fall back to a generic RELATED_TO with a `type` property.
"""


# --------------------------------------------------------------------------- #
# Cleaner approach: per-rel-type Cypher
# --------------------------------------------------------------------------- #

def _rel_cypher(rel: str, tlabel: str, tprop: str = "name") -> str:
    """Build a Cypher statement for a specific relationship type + target label."""
    return f"""
UNWIND $rows AS row
MATCH (s:Scheme {{id: row.scheme_id}})
MERGE (t:{tlabel} {{{tprop}: row.tname}})
MERGE (s)-[:{rel}]->(t)
"""


def _load_scheme(sess, sg: SchemeGraph) -> None:
    sess.run(MERGE_SCHEME, rows=[{
        "scheme_id": sg.scheme_id,
        "title": sg.title,
        "scheme_props": sg.scheme_props,
    }])

    # group triples by (rel, target_label, target_prop) and batch them
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for t in sg.triples:
        tprop = "text" if t.target_label == "Eligibility" else "name"
        key = (t.rel, t.target_label, tprop)
        by_key.setdefault(key, []).append({
            "scheme_id": sg.scheme_id,
            "tname": t.target_name,
        })

    for (rel, tlabel, tprop), rows in by_key.items():
        sess.run(_rel_cypher(rel, tlabel, tprop), rows=rows)


# --------------------------------------------------------------------------- #
# Chunks + embeddings for GraphRAG
# --------------------------------------------------------------------------- #

def _build_chunks(sg: SchemeGraph, chunk_size: int = 600,
                  overlap: int = 80) -> list[dict[str, Any]]:
    """Build text chunks for a scheme: one structured summary chunk + free-text chunks."""
    p = sg.scheme_props
    summary = (
        f"Scheme: {sg.title}\n"
        f"Department: {sg.scheme_props.get('department_id','')}\n"
        f"Funding: {p.get('funding_pattern','')}\n"
        f"How to avail: {p.get('how_to_avail','')}\n"
        f"Description: {p.get('description','')}"
    )
    chunks: list[dict[str, Any]] = []
    # split the combined text into overlapping windows
    text = summary
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append({
                "chunk_id": f"{sg.scheme_id}_c{idx}",
                "scheme_id": sg.scheme_id,
                "scheme_title": sg.title,
                "text": piece,
                "chunk_index": idx,
            })
            idx += 1
        if end >= len(text):
            break
        start = end - overlap
    return chunks


MERGE_CHUNK = """
UNWIND $rows AS row
MATCH (s:Scheme {id: row.scheme_id})
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.text = row.text,
    c.scheme_id = row.scheme_id,
    c.scheme_title = row.scheme_title,
    c.chunk_index = row.chunk_index,
    c.embedding = row.embedding
MERGE (c)-[:PART_OF]->(s)
"""


def _load_chunks(sess, sgs: list[SchemeGraph], embedder: EmbeddingsClient,
                 batch_size: int = 16) -> None:
    all_chunks: list[dict[str, Any]] = []
    for sg in sgs:
        all_chunks.extend(_build_chunks(sg))
    log.info("Built %d chunks across %d schemes", len(all_chunks), len(sgs))

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        try:
            vectors = embedder.embed([c["text"] for c in batch])
        except Exception as exc:
            log.error("Embedding batch %d failed: %s", i, exc)
            continue
        for c, vec in zip(batch, vectors, strict=True):
            c["embedding"] = vec
        sess.run(MERGE_CHUNK, rows=batch)
        log.info("Loaded %d/%d chunks", i + len(batch), len(all_chunks))


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #

def load_to_neo4j(settings: Settings, *, force: bool = False) -> None:
    """Load extracted scheme graphs (with embeddings) into Neo4j."""
    sgs = load_cached_graph(settings)

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        out("[cyan]Ensuring Neo4j schema...[/cyan]")
        _ensure_schema(driver, settings.neo4j_database, settings.embed_dimensions)

        out("[cyan]Loading scheme nodes + relationships...[/cyan]")
        with driver.session(database=settings.neo4j_database) as sess:
            for i, sg in enumerate(sgs, 1):
                _load_scheme(sess, sg)
                if i % 10 == 0 or i == len(sgs):
                    out(f"  loaded {i}/{len(sgs)} schemes")
        log.info("Loaded %d schemes into Neo4j", len(sgs))

        out("[cyan]Building chunks + embeddings...[/cyan]")
        embedder = EmbeddingsClient(settings)
        with driver.session(database=settings.neo4j_database) as sess:
            _load_chunks(sess, sgs, embedder)
        out("[green]Graph load complete.[/green]")

    finally:
        driver.close()


def get_driver(settings: Settings) -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


if __name__ == "__main__":
    from .config import get_settings

    load_to_neo4j(get_settings())
