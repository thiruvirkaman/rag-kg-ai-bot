"""Streamlit chat UI for the TN Farmers Schemes Knowledge Graph assistant.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root (parent of /app) is on sys.path so `app` is importable
# when launched via `streamlit run app/streamlit_app.py`.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from neo4j import GraphDatabase
from streamlit_agraph import Config, Edge, Node, agraph

from app.config import get_settings
from app.llm import LLMClient
from app.rag import GraphRAGRetriever, synthesize_answer

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="TN Farmers Schemes Assistant",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Tamil Nadu Farmers Schemes Assistant")
st.caption("Knowledge-Graph RAG over tn.gov.in farmer welfare schemes")


# --------------------------------------------------------------------------- #
# Sidebar: connection status + sample questions
# --------------------------------------------------------------------------- #

settings = get_settings()

with st.sidebar:
    st.header("About")
    st.markdown(
        "Ask questions about Tamil Nadu government farmer welfare schemes. "
        "Answers are grounded in a Neo4j knowledge graph built from "
        "[tn.gov.in](https://www.tn.gov.in/scheme_list.php?dep_id=Mg==)."
    )
    st.divider()
    st.header("Try asking")
    samples = [
        "What schemes are available for farmer training?",
        "Which schemes distribute seeds to farmers?",
        "How do I apply for the Soil Health Card scheme?",
        "What subsidies are available for oilseed farmers?",
        "Which schemes are sponsored by the State government?",
        "What is the funding pattern for Training to Farmers?",
    ]
    for s in samples:
        if st.button(s, key=f"sample_{s}", use_container_width=True):
            st.session_state["pending_question"] = s

    st.divider()
    st.header("Status")
    st.write(f"**LLM**: `{settings.llm_chat_model}`")
    st.write(f"**Neo4j**: `{settings.neo4j_uri}`")
    st.write(f"**Embeddings**: `{settings.embed_model}`")


# --------------------------------------------------------------------------- #
# Initialise session state
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "Hello! I can answer questions about Tamil Nadu farmer "
                "welfare schemes. Ask me about training, subsidies, seed "
                "distribution, eligibility, or how to apply."
            ),
        }
    ]
if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None


# --------------------------------------------------------------------------- #
# Lazy-init the retriever + LLM (cached on session)
# --------------------------------------------------------------------------- #

@st.cache_resource
def get_retriever():
    return GraphRAGRetriever(get_settings())


@st.cache_resource
def get_llm():
    return LLMClient(get_settings())


@st.cache_resource
def get_neo4j_driver():
    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


# --------------------------------------------------------------------------- #
# Graph visualisation helpers
# --------------------------------------------------------------------------- #

NODE_COLORS = {
    "Scheme": "#2E86AB",
    "Department": "#A23B72",
    "Sponsor": "#F18F01",
    "Beneficiary": "#C73E1D",
    "BenefitType": "#6A994E",
    "Crop": "#386641",
    "Input": "#BC4749",
    "Activity": "#577590",
    "District": "#F2CC8F",
    "ContactRole": "#81B29A",
    "Eligibility": "#E07A5F",
    "Chunk": "#D3D3D3",
}

NODE_SIZES = {
    "Scheme": 35,
    "Department": 25,
    "Sponsor": 25,
    "Beneficiary": 22,
    "BenefitType": 22,
    "Crop": 20,
    "Input": 20,
    "Activity": 20,
    "District": 18,
    "ContactRole": 18,
    "Eligibility": 18,
    "Chunk": 12,
}


def _node_id(label: str, name: str) -> str:
    return f"{label}::{name}"


def _build_agraph(nodes_data, edges_data, height: int = 600):
    """Build an interactive graph from node/edge lists."""
    nodes = []
    for n in nodes_data:
        label = n["label"]
        name = n["name"]
        color = NODE_COLORS.get(label, "#888888")
        size = NODE_SIZES.get(label, 20)
        nodes.append(Node(
            id=_node_id(label, name),
            label=name if len(name) <= 30 else name[:27] + "...",
            size=size,
            color=color,
            title=f"{label}: {name}",
        ))
    edges = []
    for e in edges_data:
        edges.append(Edge(
            source=_node_id(e["source_label"], e["source_name"]),
            target=_node_id(e["target_label"], e["target_name"]),
            label=e["rel"].replace("_", " "),
            color="#999999",
        ))
    config = Config(
        width="100%",
        height=height,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F18F01",
        collapsible=True,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": False},
    )
    return agraph(nodes=nodes, edges=edges, config=config)


def _fetch_all_schemes_list(driver) -> list[dict]:
    """Return list of {id, title} for the scheme dropdown."""
    with driver.session(database=settings.neo4j_database) as sess:
        result = sess.run(
            "MATCH (s:Scheme) RETURN s.id AS id, s.title AS title "
            "ORDER BY s.title"
        )
        return [{"id": r["id"], "title": r["title"]} for r in result]


def _fetch_subgraph(driver, scheme_id: str, hops: int = 1) -> tuple[list, list]:
    """Fetch a scheme's subgraph (nodes + edges) up to N hops."""
    cypher = """
    MATCH (s:Scheme {id: $sid})
    OPTIONAL MATCH (s)-[r*1..""" + str(hops) + """]-(t)
    WITH s, collect(DISTINCT r) AS all_paths
    UNWIND all_paths AS path
    UNWIND path AS rel
    WITH collect(DISTINCT rel) AS rels
    UNWIND rels AS rel
    RETURN startNode(rel) AS sn, type(rel) AS rel_type, endNode(rel) AS en,
           labels(startNode(rel))[0] AS sn_label,
           labels(endNode(rel))[0] AS en_label
    """
    with driver.session(database=settings.neo4j_database) as sess:
        records = sess.run(cypher, sid=scheme_id).data()

    nodes_map: dict[str, dict] = {}
    edges: list[dict] = []
    for rec in records:
        sn_label = rec["sn_label"]
        en_label = rec["en_label"]
        sn_name = rec["sn"].get("title") or rec["sn"].get("name") or rec["sn"].get("text", "")
        en_name = rec["en"].get("title") or rec["en"].get("name") or rec["en"].get("text", "")
        if not sn_name or not en_name:
            continue
        # skip Chunk nodes (too noisy in visualisation)
        if sn_label == "Chunk" or en_label == "Chunk":
            continue
        sn_key = _node_id(sn_label, sn_name)
        en_key = _node_id(en_label, en_name)
        if sn_key not in nodes_map:
            nodes_map[sn_key] = {"label": sn_label, "name": sn_name}
        if en_key not in nodes_map:
            nodes_map[en_key] = {"label": en_label, "name": en_name}
        edges.append({
            "source_label": sn_label,
            "source_name": sn_name,
            "target_label": en_label,
            "target_name": en_name,
            "rel": rec["rel_type"],
        })
    return list(nodes_map.values()), edges


def _fetch_overview_graph(driver, limit: int = 150) -> tuple[list, list]:
    """Fetch a high-level overview: Schemes connected to their key entities."""
    cypher = """
    MATCH (s:Scheme)-[r]->(t)
    WHERE NOT labels(t)[0] = 'Chunk'
    RETURN s.title AS s_title, labels(t)[0] AS t_label,
           coalesce(t.name, t.text) AS t_name, type(r) AS rel
    LIMIT $limit
    """
    with driver.session(database=settings.neo4j_database) as sess:
        records = sess.run(cypher, limit=limit).data()

    nodes_map: dict[str, dict] = {}
    edges: list[dict] = []
    for rec in records:
        s_name = rec["s_title"]
        t_name = rec["t_name"]
        t_label = rec["t_label"]
        if not s_name or not t_name:
            continue
        s_key = _node_id("Scheme", s_name)
        t_key = _node_id(t_label, t_name)
        if s_key not in nodes_map:
            nodes_map[s_key] = {"label": "Scheme", "name": s_name}
        if t_key not in nodes_map:
            nodes_map[t_key] = {"label": t_label, "name": t_name}
        edges.append({
            "source_label": "Scheme",
            "source_name": s_name,
            "target_label": t_label,
            "target_name": t_name,
            "rel": rec["rel"],
        })
    return list(nodes_map.values()), edges


def _fetch_stats(driver) -> dict:
    """Fetch node counts per label for the overview."""
    with driver.session(database=settings.neo4j_database) as sess:
        result = sess.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count "
            "ORDER BY count DESC"
        )
        return {r["label"]: r["count"] for r in result}


# --------------------------------------------------------------------------- #
# Tabs: Chat + Graph Explorer
# --------------------------------------------------------------------------- #

tab_chat, tab_graph = st.tabs(["💬 Chat", "🕸️ Graph Explorer"])

# --------------------------------------------------------------------------- #
# Tab 1: Chat
# --------------------------------------------------------------------------- #

with tab_chat:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        title = src.get("title", "(untitled)")
                        url = src.get("url", "")
                        score = src.get("score", 0.0)
                        if url:
                            st.markdown(f"- [{title}]({url})  _(score: {score:.3f})_")
                        else:
                            st.markdown(f"- {title}  _(score: {score:.3f})_")

    def handle_question(question: str) -> None:
        """Run retrieval + synthesis for a question and append to history."""
        st.session_state["messages"].append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge graph..."):
                try:
                    retriever = get_retriever()
                    llm = get_llm()
                    contexts = retriever.retrieve(question)
                    answer = synthesize_answer(question, contexts, llm)
                    sources = [
                        {"title": c.title, "url": c.url, "score": c.score}
                        for c in contexts
                    ]
                except Exception as exc:
                    answer = (
                        f"Sorry, I could not answer right now. "
                        f"Error: {exc}"
                    )
                    sources = []
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        title = src.get("title", "(untitled)")
                        url = src.get("url", "")
                        score = src.get("score", 0.0)
                        if url:
                            st.markdown(
                                f"- [{title}]({url})  _(score: {score:.3f})_"
                            )
                        else:
                            st.markdown(f"- {title}  _(score: {score:.3f})_")

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    if st.session_state.get("pending_question"):
        q = st.session_state["pending_question"]
        st.session_state["pending_question"] = None
        handle_question(q)
        st.rerun()

    user_input = st.chat_input("Ask about Tamil Nadu farmer schemes...")
    if user_input:
        handle_question(user_input)


# --------------------------------------------------------------------------- #
# Tab 2: Graph Explorer
# --------------------------------------------------------------------------- #

with tab_graph:
    st.subheader("Knowledge Graph Visualisation")
    st.markdown(
        "Explore the Neo4j knowledge graph interactively. "
        "Drag nodes to rearrange, hover for details, scroll to zoom."
    )

    try:
        driver = get_neo4j_driver()
    except Exception as exc:
        st.error(f"Cannot connect to Neo4j: {exc}")
        driver = None

    if driver:
        # --- Graph stats ---
        stats = _fetch_stats(driver)
        cols = st.columns(len(stats))
        for col, (label, count) in zip(cols, stats.items(), strict=False):
            color = NODE_COLORS.get(label, "#888888")
            col.metric(label, count)
            col.markdown(
                f'<div style="width:12px;height:12px;background:{color};'
                f'border-radius:50%;display:inline-block;margin-left:4px;"></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # --- View mode selector ---
        view_mode = st.radio(
            "View mode",
            ["Overview (all schemes)", "Single scheme subgraph"],
            horizontal=True,
            help="Overview shows all schemes and their key entities. "
                 "Single scheme shows the full neighbourhood of one scheme.",
        )

        if view_mode == "Overview (all schemes)":
            with st.spinner("Loading graph overview..."):
                nodes_data, edges_data = _fetch_overview_graph(driver, limit=200)
            st.caption(
                f"Showing {len(nodes_data)} nodes and {len(edges_data)} "
                f"relationships (limited to 200 for performance)."
            )
            _build_agraph(nodes_data, edges_data, height=650)

        else:
            # Single scheme subgraph
            schemes = _fetch_all_schemes_list(driver)
            scheme_titles = [s["title"] for s in schemes]
            selected_title = st.selectbox(
                "Select a scheme to explore:",
                scheme_titles,
                index=0,
            )
            selected_id = next(
                s["id"] for s in schemes if s["title"] == selected_title
            )
            hops = st.slider("Hops (traversal depth)", 1, 3, 1,
                             help="How many relationships to follow from the scheme.")

            with st.spinner(f"Loading subgraph for '{selected_title}'..."):
                nodes_data, edges_data = _fetch_subgraph(driver, selected_id, hops)

            st.caption(
                f"Showing {len(nodes_data)} nodes and {len(edges_data)} "
                f"relationships for **{selected_title}**."
            )
            if nodes_data:
                _build_agraph(nodes_data, edges_data, height=650)
            else:
                st.info("No relationships found for this scheme.")

        # --- Legend ---
        st.divider()
        st.markdown("**Legend:**")
        legend_cols = st.columns(6)
        shown = 0
        for label, color in NODE_COLORS.items():
            if label == "Chunk":
                continue
            col = legend_cols[shown % 6]
            col.markdown(
                f'<span style="color:{color};font-size:18px;">●</span> {label}',
                unsafe_allow_html=True,
            )
            shown += 1
