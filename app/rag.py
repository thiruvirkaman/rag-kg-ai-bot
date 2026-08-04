"""GraphRAG retrieval + answer synthesis.

Pipeline step 5: Graph Query.

Retrieval strategy (hybrid):
  1. Embed the user question and run a vector search over ``:Chunk`` nodes
     to find the top-k semantically relevant scheme chunks.
  2. From each matched scheme, traverse N hops out in the graph to collect
     its neighbouring entities (Department, Sponsor, Beneficiary, Benefit,
     Crops, Inputs, Districts, Eligibility, ContactRoles, etc.).
  3. Build a structured "context block" per scheme containing both the
     chunk text and the graph subgraph facts.
  4. Feed the assembled context + the user question to the LLM to produce a
     grounded answer with citations to the scheme title + URL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neo4j import Driver

from .config import Settings
from .console import get_logger
from .embeddings import EmbeddingsClient
from .graph_store import get_driver
from .llm import LLMClient

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Cypher: vector search over chunks
# --------------------------------------------------------------------------- #

VECTOR_SEARCH = """
CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
YIELD node, score
RETURN node.chunk_id      AS chunk_id,
       node.scheme_id     AS scheme_id,
       node.scheme_title  AS scheme_title,
       node.text          AS text,
       node.chunk_index   AS chunk_index,
       score
ORDER BY score DESC
"""


# --------------------------------------------------------------------------- #
# Cypher: subgraph traversal from a scheme (2 hops)
# --------------------------------------------------------------------------- #

SUBGRAPH = """
MATCH (s:Scheme {id: $scheme_id})
OPTIONAL MATCH (s)-[r]->(t)
RETURN s.id                  AS scheme_id,
       s.title               AS title,
       s.url                 AS url,
       s.funding_pattern     AS funding_pattern,
       s.how_to_avail        AS how_to_avail,
       s.description         AS description,
       s.scheme_type         AS scheme_type,
       collect(DISTINCT {
         rel: type(r),
         target_label: labels(t)[0],
         target_name: coalesce(t.name, t.text)
       }) AS neighbours
"""


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #

@dataclass
class SchemeContext:
    scheme_id: str
    title: str
    url: str
    chunks: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    score: float = 0.0

    def context_block(self) -> str:
        parts = [f"### Scheme: {self.title}"]
        parts.append(f"Source: {self.url}")
        if self.facts:
            parts.append("Facts:")
            for f in self.facts:
                parts.append(f"  - {f}")
        if self.chunks:
            parts.append("Excerpt:")
            for c in self.chunks:
                parts.append(f"  > {c}")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #

class GraphRAGRetriever:
    def __init__(self, settings: Settings, driver: Driver | None = None,
                 embedder: EmbeddingsClient | None = None):
        self.settings = settings
        self.driver = driver or get_driver(settings)
        self.embedder = embedder or EmbeddingsClient(settings)

    # -- public ------------------------------------------------------------ #

    def retrieve(self, question: str) -> list[SchemeContext]:
        """Return the top-k scheme contexts for the question."""
        q_emb = self.embedder.embed_one(question)
        with self.driver.session(database=self.settings.neo4j_database) as sess:
            hits = sess.run(VECTOR_SEARCH,
                            k=self.settings.retrieval_top_k,
                            embedding=q_emb).data()
        if not hits:
            return []

        # group chunks by scheme_id
        by_scheme: dict[str, SchemeContext] = {}
        for h in hits:
            sid = h["scheme_id"]
            ctx = by_scheme.get(sid)
            if ctx is None:
                ctx = SchemeContext(
                    scheme_id=sid,
                    title=h.get("scheme_title", ""),
                    url="",
                    score=float(h.get("score", 0.0)),
                )
                by_scheme[sid] = ctx
            ctx.chunks.append(h["text"])

        # fetch subgraph facts for each matched scheme
        with self.driver.session(database=self.settings.neo4j_database) as sess:
            for sid, ctx in by_scheme.items():
                rec = sess.run(SUBGRAPH, scheme_id=sid).single()
                if rec is None:
                    continue
                ctx.title = rec["title"]
                ctx.url = rec["url"] or ""
                ctx.facts = self._facts_from_neighbours(rec, sess)

        # rank by combined vector score + number of facts
        ranked = sorted(by_scheme.values(),
                        key=lambda c: (c.score, len(c.facts)), reverse=True)
        return ranked

    # -- helpers ----------------------------------------------------------- #

    def _facts_from_neighbours(self, rec, sess) -> list[str]:
        facts: list[str] = []
        # structured fields directly on the scheme node
        if rec.get("funding_pattern"):
            facts.append(f"Funding pattern: {rec['funding_pattern']}")
        if rec.get("how_to_avail"):
            facts.append(f"How to avail: {rec['how_to_avail']}")
        if rec.get("scheme_type"):
            facts.append(f"Scheme type: {rec['scheme_type']}")

        neighbours = rec.get("neighbours") or []
        # group by relationship type for readable facts
        by_rel: dict[str, list[str]] = {}
        for n in neighbours:
            if not n or not n.get("target_name"):
                continue
            rel = n["rel"]
            label = n.get("target_label", "")
            by_rel.setdefault(rel, []).append(
                f"{label}:{n['target_name']}" if label else n["target_name"]
            )

        rel_english = {
            "OFFERED_BY": "Sponsored by",
            "RUN_BY": "Run by",
            "TARGETS": "Beneficiaries",
            "PROVIDES": "Provides",
            "SUPPORTS_CROP": "Supports crops",
            "DISTRIBUTES": "Distributes",
            "INCLUDES_ACTIVITY": "Includes activity",
            "APPLICABLE_IN": "Applicable in districts",
            "APPLY_TO": "Apply to",
            "HAS_ELIGIBILITY": "Eligibility",
            "PART_OF": "Part of",
        }
        for rel, names in by_rel.items():
            if rel == "PART_OF":
                continue
            label = rel_english.get(rel, rel.replace("_", " ").title())
            facts.append(f"{label}: {', '.join(names)}")
        return facts


# --------------------------------------------------------------------------- #
# Answer synthesis
# --------------------------------------------------------------------------- #

ANSWER_PROMPT = """You are an assistant that answers questions about Tamil Nadu government farmer welfare schemes. Use ONLY the context below. If the context does not contain the answer, say you don't have enough information rather than guessing.

For each scheme you reference in the answer, cite it as: [Scheme: <title>].

Be precise, concise, and accurate. Group related schemes together when helpful.

Context:
{context}

Question: {question}

Answer:"""


def synthesize_answer(question: str, contexts: list[SchemeContext],
                      llm: LLMClient) -> str:
    if not contexts:
        return ("I don't have enough information to answer that. "
                "Try asking about a specific farmer scheme (e.g. subsidies, "
                "training, seed distribution).")
    context_text = "\n\n".join(c.context_block() for c in contexts[:6])
    prompt = ANSWER_PROMPT.format(context=context_text, question=question)
    return llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=llm.settings.llm_temperature,
    ).strip()


# --------------------------------------------------------------------------- #
# End-to-end Q&A
# --------------------------------------------------------------------------- #

def answer_question(question: str, settings: Settings) -> dict[str, Any]:
    """Run the full retrieve-then-generate pipeline for a question."""
    llm = LLMClient(settings)
    retriever = GraphRAGRetriever(settings)
    contexts = retriever.retrieve(question)
    answer = synthesize_answer(question, contexts, llm)
    return {
        "answer": answer,
        "sources": [
            {"title": c.title, "url": c.url, "score": c.score}
            for c in contexts
        ],
    }
