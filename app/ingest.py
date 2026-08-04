"""Ingestion orchestrator.

Runs the full pipeline end-to-end:
    1. Data Load        (app.scrape)      -> data/schemes_raw.json
    2. Entity Extract   (app.extract)     -> data/schemes_chunks.json
    3. Graph Storage    (app.graph_store) -> Neo4j

Usage:
    python -m app.ingest            # run all steps (uses cache if present)
    python -m app.ingest --force    # re-run every step from scratch
    python -m app.ingest --no-llm   # extraction without LLM (rules only)
"""
from __future__ import annotations

import argparse

from .config import get_settings
from .console import die, get_logger, out
from .extract import extract_all
from .graph_store import load_to_neo4j
from .scrape import scrape

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG-KG ingest pipeline.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every step, ignoring caches.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Entity extraction without LLM (rules only).")
    parser.add_argument("--skip-load", action="store_true",
                        help="Skip the Neo4j load step.")
    args = parser.parse_args()

    settings = get_settings()

    # --- Step 1: Data Load ------------------------------------------------ #
    out("[bold cyan]Step 1/3: Data Load (scrape)[/bold cyan]")
    try:
        scrape(settings, force=args.force)
    except Exception as exc:
        die(f"Scrape failed: {exc}")

    # --- Step 2: Entity Extraction + Relationship Mapping ----------------- #
    out("[bold cyan]Step 2/3: Entity Extraction + Relationship Mapping[/bold cyan]")
    try:
        extract_all(settings, use_llm=not args.no_llm, force=args.force)
    except Exception as exc:
        die(f"Extraction failed: {exc}")

    # --- Step 3: Graph Storage ------------------------------------------- #
    if args.skip_load:
        out("[yellow]Skipping Neo4j load (--skip-load).[/yellow]")
    else:
        out("[bold cyan]Step 3/3: Graph Storage (Neo4j)[/bold cyan]")
        try:
            load_to_neo4j(settings, force=args.force)
        except Exception as exc:
            die(f"Neo4j load failed: {exc}")

    out("[bold green]Ingest pipeline complete.[/bold green]")


if __name__ == "__main__":
    main()
