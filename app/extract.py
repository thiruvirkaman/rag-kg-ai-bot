"""Entity extraction & relationship mapping.

Pipeline step 2 + 3: extract entities from each scheme's free-text fields
(Description, How To Avail, Funding Pattern) and map them into structured
``(Entity)-[RELATIONSHIP]->(Entity)`` triples that will be loaded into Neo4j.

We combine:
  * Rule-based extraction of the structured fields (Department, Sponsor,
    Beneficiary, Benefit Type, Districts, etc.) -- deterministic.
  * LLM-based extraction of domain entities (crops, inputs, activities,
    districts, eligibility constraints, application contacts) from the
    free-text Description / How To Avail.

Schema (node labels / relationship types):

  (:Scheme {id, title, ...})
  (:Department {name})
  (:Sponsor {name})              -- State / Central / Both
  (:Beneficiary {name})          -- Farmers / Women / SC/ST / ...
  (:BenefitType {name})          -- Subsidy / Grant / Incentive / ...
  (:Crop {name})
  (:Input {name})                -- Seeds / Gypsum / Rhizobium / ...
  (:Activity {name})             -- Training / Demonstration / ...
  (:District {name})
  (:ContactRole {name})          -- Assistant Agricultural Officer / ...
  (:Eligibility {text})

  (Scheme)-[:OFFERED_BY]->(Sponsor)
  (Scheme)-[:RUN_BY]->(Department)
  (Scheme)-[:TARGETS]->(Beneficiary)
  (Scheme)-[:PROVIDES]->(BenefitType)
  (Scheme)-[:SUPPORTS_CROP]->(Crop)
  (Scheme)-[:DISTRIBUTES]->(Input)
  (Scheme)-[:INCLUDES_ACTIVITY]->(Activity)
  (Scheme)-[:APPLICABLE_IN]->(District)
  (Scheme)-[:APPLY_TO]->(ContactRole)
  (Scheme)-[:HAS_ELIGIBILITY]->(Eligibility)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .console import get_logger, out
from .llm import LLMClient
from .scrape import Scheme, load_cached

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Canonical node types
# --------------------------------------------------------------------------- #

NODE_LABELS = (
    "Scheme", "Department", "Sponsor", "Beneficiary", "BenefitType",
    "Crop", "Input", "Activity", "District", "ContactRole", "Eligibility",
)

REL_TYPES = (
    "OFFERED_BY", "RUN_BY", "TARGETS", "PROVIDES", "SUPPORTS_CROP",
    "DISTRIBUTES", "INCLUDES_ACTIVITY", "APPLICABLE_IN", "APPLY_TO",
    "HAS_ELIGIBILITY",
)


# --------------------------------------------------------------------------- #
# Data model for extraction output
# --------------------------------------------------------------------------- #

@dataclass
class Triple:
    source_label: str
    source_name: str
    rel: str
    target_label: str
    target_name: str
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemeGraph:
    scheme_id: str
    title: str
    scheme_props: dict[str, Any]
    triples: list[Triple] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme_id": self.scheme_id,
            "title": self.title,
            "scheme_props": self.scheme_props,
            "triples": [
                {
                    "source_label": t.source_label,
                    "source_name": t.source_name,
                    "rel": t.rel,
                    "target_label": t.target_label,
                    "target_name": t.target_name,
                    "props": t.props,
                }
                for t in self.triples
            ],
        }


# --------------------------------------------------------------------------- #
# Rule-based extraction over structured fields
# --------------------------------------------------------------------------- #

_TN_DISTRICTS = {
    "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore",
    "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kancheepuram",
    "Kanniyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
    "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai",
    "Ramanathapuram", "Ranipet", "Salem", "Sivagangai", "Tenkasi", "Thanjavur",
    "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli", "Tirupathur",
    "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur", "Vellore",
    "Viluppuram", "Virudhunagar",
    # common spelling variants seen on tn.gov.in
    "Trichy", "Tuticorin", "Kanyakumari", "Thirunelveli",
}

_DISTRICT_ALIASES = {
    "Trichy": "Tiruchirappalli",
    "Tuticorin": "Thoothukudi",
    "Kanyakumari": "Kanniyakumari",
    "Thirunelveli": "Tirunelveli",
}


def _split_list(value: str) -> list[str]:
    """Split a comma/slash/and separated list into clean items."""
    if not value:
        return []
    parts = re.split(r"[,/;]|\band\b", value, flags=re.IGNORECASE)
    out = []
    for p in parts:
        p = p.strip().strip(".")
        if p and p.lower() not in ("na", "n/a", "not applicable", "-"):
            out.append(p)
    return out


def _extract_districts(text: str) -> list[str]:
    found: list[str] = []
    for d in _TN_DISTRICTS:
        if re.search(r"\b" + re.escape(d) + r"\b", text, re.IGNORECASE):
            canon = _DISTRICT_ALIASES.get(d, d)
            if canon not in found:
                found.append(canon)
    return found


_CONTACT_ROLE_RE = re.compile(
    r"(Assistant Agricultural Officer|Deputy Agricultural Officer|"
    r"Assistant Director of Agriculture|Joint Director of Agriculture|"
    r"Assistant Director of Agriculture|Block Level|District Level|"
    r"Village Level|Agricultural Officer)",
    re.IGNORECASE,
)


def _extract_contact_roles(text: str) -> list[str]:
    if not text:
        return []
    roles: list[str] = []
    for m in _CONTACT_ROLE_RE.finditer(text):
        role = m.group(1).title()
        if role not in roles:
            roles.append(role)
    return roles


# --------------------------------------------------------------------------- #
# LLM-based extraction over free-text
# --------------------------------------------------------------------------- #

EXTRACT_PROMPT = """You are an information-extraction assistant for Tamil Nadu government farmer welfare schemes.

From the scheme text below, extract domain entities and relationships.
Return ONLY a JSON object with this exact shape:

{
  "crops": ["string", ...],            // crops the scheme supports (maize, paddy, pulses, oilseeds, groundnut, cotton, millets, ...)
  "inputs": ["string", ...],           // agricultural inputs distributed (seeds, gypsum, rhizobium, micro nutrients, plant protection equipment, pipes, minikits, ...)
  "activities": ["string", ...],       // activities conducted (training, demonstration, seed production, distribution, ...)
  "districts": ["string", ...],        // Tamil Nadu districts where the scheme applies (use canonical district names)
  "contact_roles": ["string", ...],    // officer/official roles to apply to (e.g. Assistant Agricultural Officer)
  "eligibility": ["string", ...]       // short eligibility constraints stated in the text (max ~12 words each)
}

Rules:
- Use canonical names (title case). E.g. "Paddy" not "paddy".
- Only include entities explicitly mentioned in the text. Do NOT invent.
- If a category is empty, return an empty list.
- Do not duplicate entities. Be precise.
- Output ONLY the JSON object, no prose.

Scheme title: {title}

Scheme text:
\"\"\"
{text}
\"\"\"
"""


def _llm_extract(llm: LLMClient, scheme: Scheme) -> dict[str, list[str]]:
    text_parts = []
    for f in ("description", "how_to_avail", "funding_pattern",
              "benefit_type", "beneficiaries"):
        v = getattr(scheme, f, "").strip()
        if v:
            text_parts.append(f"{f.replace('_',' ').title()}: {v}")
    text = "\n".join(text_parts)
    if not text.strip():
        return {k: [] for k in
                ("crops", "inputs", "activities", "districts",
                 "contact_roles", "eligibility")}

    prompt = EXTRACT_PROMPT.replace("{title}", scheme.title).replace("{text}", text)
    data = llm.chat_json(
        [{"role": "user", "content": prompt}],
        model=llm.settings.llm_extract_model,
        temperature=0.0,
    )
    # validate shape
    out: dict[str, list[str]] = {}
    for key in ("crops", "inputs", "activities", "districts",
                "contact_roles", "eligibility"):
        val = data.get(key, [])
        if not isinstance(val, list):
            val = []
        out[key] = [str(x).strip() for x in val if str(x).strip()]
    return out


# --------------------------------------------------------------------------- #
# Combine rule-based + LLM into triples
# --------------------------------------------------------------------------- #

def build_scheme_graph(scheme: Scheme, llm: LLMClient | None) -> SchemeGraph:
    """Build the graph (triples) for a single scheme."""
    sg = SchemeGraph(
        scheme_id=scheme.scheme_id,
        title=scheme.title,
        scheme_props={
            "id": scheme.scheme_id,
            "title": scheme.title,
            "url": scheme.url,
            "funding_pattern": scheme.funding_pattern,
            "introduced_on": scheme.introduced_on,
            "scheme_type": scheme.scheme_type,
            "associated_scheme": scheme.associated_scheme,
            "uploaded_file": scheme.uploaded_file,
            "description": scheme.description,
            "how_to_avail": scheme.how_to_avail,
            "organisation": scheme.organisation,
            "department_id": scheme.department_id,
        },
    )

    def add(rel: str, tlabel: str, tname: str, slabel: str = "Scheme",
            sname: str | None = None, props: dict | None = None):
        sname = sname if sname is not None else scheme.title
        sg.triples.append(Triple(slabel, sname, rel, tlabel, tname,
                                 props or {}))

    # --- Rule-based: structured fields ---
    if scheme.department:
        add("RUN_BY", "Department", scheme.department)
    if scheme.sponsored_by:
        add("OFFERED_BY", "Sponsor", scheme.sponsored_by.strip())
    for b in _split_list(scheme.beneficiaries):
        add("TARGETS", "Beneficiary", b)
    for bt in _split_list(scheme.benefit_type):
        add("PROVIDES", "BenefitType", bt)

    # --- Rule-based: districts / contacts from how_to_avail + description ---
    rule_text = f"{scheme.how_to_avail}\n{scheme.description}"
    for d in _extract_districts(rule_text):
        add("APPLICABLE_IN", "District", d)
    for r in _extract_contact_roles(scheme.how_to_avail):
        add("APPLY_TO", "ContactRole", r)

    # --- LLM-based: free-text entity extraction ---
    if llm is not None:
        try:
            ent = _llm_extract(llm, scheme)
        except Exception as exc:
            log.error("LLM extraction failed for %s: %s", scheme.scheme_id, exc)
            ent = {k: [] for k in ("crops", "inputs", "activities",
                                   "districts", "contact_roles", "eligibility")}

        for c in ent["crops"]:
            add("SUPPORTS_CROP", "Crop", c)
        for inp in ent["inputs"]:
            add("DISTRIBUTES", "Input", inp)
        for act in ent["activities"]:
            add("INCLUDES_ACTIVITY", "Activity", act)
        # merge LLM districts with rule-based (dedupe)
        existing_d = {t.target_name for t in sg.triples
                      if t.rel == "APPLICABLE_IN"}
        for d in ent["districts"]:
            canon = _DISTRICT_ALIASES.get(d, d)
            if canon not in existing_d:
                add("APPLICABLE_IN", "District", canon)
                existing_d.add(canon)
        existing_r = {t.target_name for t in sg.triples
                      if t.rel == "APPLY_TO"}
        for r in ent["contact_roles"]:
            if r not in existing_r:
                add("APPLY_TO", "ContactRole", r)
                existing_r.add(r)
        for el in ent["eligibility"]:
            add("HAS_ELIGIBILITY", "Eligibility", el)
    else:
        # no LLM: still derive a coarse eligibility from description
        if scheme.description.strip():
            add("HAS_ELIGIBILITY", "Eligibility", scheme.description.strip())

    return sg


# --------------------------------------------------------------------------- #
# Public pipeline
# --------------------------------------------------------------------------- #

def extract_all(
    settings: Settings,
    *,
    use_llm: bool = True,
    force: bool = False,
) -> list[SchemeGraph]:
    """Extract entities/relationships for every cached scheme.

    Results are cached to ``settings.chunks_cache``.
    """
    cache_path = Path(settings.chunks_cache)
    if cache_path.exists() and not force:
        log.info("Loading cached graph extraction from %s", cache_path)
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [_dict_to_scheme_graph(d) for d in data]

    schemes = load_cached(settings)
    llm = None
    if use_llm:
        from .llm import make_client
        llm = make_client(settings)

    out_list: list[SchemeGraph] = []
    total = len(schemes)
    for i, scheme in enumerate(schemes, 1):
        out(f"  [{i}/{total}] extracting {scheme.scheme_id} - {scheme.title[:40]}")
        sg = build_scheme_graph(scheme, llm)
        out_list.append(sg)
        log.info("Extracted %s: %d triples", scheme.scheme_id, len(sg.triples))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([sg.to_dict() for sg in out_list], indent=2,
                   ensure_ascii=False),
        encoding="utf-8",
    )
    out(f"[green]Cached {len(out_list)} scheme graphs ->[/green] {cache_path}")
    return out_list


def _dict_to_scheme_graph(d: dict[str, Any]) -> SchemeGraph:
    return SchemeGraph(
        scheme_id=d["scheme_id"],
        title=d["title"],
        scheme_props=d["scheme_props"],
        triples=[Triple(**t) for t in d["triples"]],
    )


def load_cached_graph(settings: Settings) -> list[SchemeGraph]:
    cache_path = Path(settings.chunks_cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached graph at {cache_path}. Run `python -m app.extract`."
        )
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return [_dict_to_scheme_graph(d) for d in data]


if __name__ == "__main__":
    from .config import get_settings

    extract_all(get_settings())
