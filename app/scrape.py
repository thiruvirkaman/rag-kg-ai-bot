"""Scrape Tamil Nadu government farmer schemes from tn.gov.in.

Pipeline step 1: Data Load.

The listing page (scheme_list.php?dep_id=Mg==) contains links to individual
scheme detail pages. Each detail page has labelled fields rendered as
``<b>Label:</b> value``. We parse every field deterministically and also
keep the raw description / how-to-avail text for downstream LLM extraction.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag

from .config import Settings
from .console import get_logger, out

log = get_logger(__name__)

# Field labels we explicitly capture from each scheme detail page.
# (display_label, internal_key)
KNOWN_FIELDS: list[tuple[str, str]] = [
    ("Concerned Department", "department"),
    ("Concerned District", "district"),
    ("Organisation Name", "organisation"),
    ("Scheme Title/Name", "title"),
    ("Associated Scheme", "associated_scheme"),
    ("Sponsered By", "sponsored_by"),
    ("Funding Pattern", "funding_pattern"),
    ("Beneficiaries", "beneficiaries"),
    ("Types of Benefits", "benefit_type"),
    ("Income", "income"),
    ("Age From", "age_from"),
    ("Age To", "age_to"),
    ("Community", "community"),
    ("How To avail", "how_to_avail"),
    ("Introduced On", "introduced_on"),
    ("Description", "description"),
    ("Scheme Type", "scheme_type"),
    ("Uploaded File", "uploaded_file"),
]

# Section headers on the detail page that group known fields.
SECTION_HEADERS = {
    "Scheme Details:",
    "Eligibility criteria:",
    "Validity of the Scheme:",
}


@dataclass
class Scheme:
    """A single scheme record, serialisable to JSON."""

    scheme_id: str = ""
    url: str = ""
    department: str = ""
    title: str = ""
    sponsored_by: str = ""
    funding_pattern: str = ""
    beneficiaries: str = ""
    benefit_type: str = ""
    income: str = ""
    age_from: str = ""
    age_to: str = ""
    community: str = ""
    how_to_avail: str = ""
    introduced_on: str = ""
    description: str = ""
    scheme_type: str = ""
    associated_scheme: str = ""
    district: str = ""
    organisation: str = ""
    uploaded_file: str = ""
    department_id: str = ""
    scraped_at: str = ""

    # extra/unknown labelled fields the page may add later
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "en-IN,en;q=0.9",
        }
    )
    return s


def _get(session: requests.Session, url: str, retries: int = 3) -> str:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            # The site is utf-8; requests may mis-detect. Force it.
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as exc:
            last_err = exc
            wait = 2 * attempt
            log.warning("GET %s failed (attempt %d/%d): %s -- retrying in %ds",
                        url, attempt, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err}")


# --------------------------------------------------------------------------- #
# Listing parse
# --------------------------------------------------------------------------- #

_SCHEME_HREF_RE = re.compile(r"^scheme_details\.php\?id=([^&]+)$")


def parse_scheme_links(html: str, base: str) -> list[tuple[str, str]]:
    """Return list of (scheme_id, absolute_url) from the listing page."""
    soup = BeautifulSoup(html, "lxml")
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        m = _SCHEME_HREF_RE.match(a["href"].strip())
        if not m:
            continue
        sid = m.group(1)
        if sid in seen:
            continue
        seen.add(sid)
        url = base.rstrip("/") + "/scheme_details.php?id=" + sid
        links.append((sid, url))
    return links


# --------------------------------------------------------------------------- #
# Detail parse
# --------------------------------------------------------------------------- #

def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _row_fields(tr: Tag) -> tuple[str, str]:
    """Extract (label, value) from a <tr> with <td><b>Label:</b></td><td>v</td>.

    Returns ("", "") if the row is not a labelled field row.
    """
    tds = tr.find_all("td", recursive=False)
    if len(tds) < 2:
        return "", ""
    label_cell = tds[0]
    value_cell = tds[1]

    b = label_cell.find("b")
    if b is None:
        return "", ""
    label = _clean(b.get_text(" ", strip=True).rstrip(":"))
    if not label:
        return "", ""

    # value = text of the value cell, excluding any nested <b> (rare)
    # strip nested <b> to avoid picking up the next field label
    for nested_b in value_cell.find_all("b"):
        nested_b.extract()
    value = _clean(value_cell.get_text(" ", strip=True))
    return label, value


def parse_scheme_detail(html: str, scheme_id: str, url: str,
                        department_id: str) -> Scheme:
    """Parse a scheme_details.php page into a Scheme dataclass.

    The page uses a <table> where each row is ``<tr><td><b>Label:</b></td>
    <td>value</td></tr>``. We walk every <tr> in the document, extract
    label/value pairs, and map known labels to Scheme fields.
    """
    soup = BeautifulSoup(html, "lxml")
    scheme = Scheme(
        scheme_id=scheme_id,
        url=url,
        department_id=department_id,
        scraped_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    label_to_key = dict(KNOWN_FIELDS)

    for tr in soup.find_all("tr"):
        label, value = _row_fields(tr)
        if not label or not value:
            continue
        key = label_to_key.get(label)
        if key:
            setattr(scheme, key, value)
        else:
            # keep unknown labelled fields for completeness
            scheme.extra[label] = value

    # Fallback: if title wasn't found, try the page heading (h1-h4),
    # skipping generic "Schemes" / "Government of Tamil Nadu" headers.
    if not scheme.title:
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            t = _clean(h.get_text(" ", strip=True))
            if t and t.lower() not in ("schemes", "government of tamil nadu",
                                       "accessibility menu"):
                scheme.title = t
                break

    # Fallback: department from breadcrumb if missing
    if not scheme.department:
        scheme.department = "Agriculture - Farmers Welfare Department"

    return scheme


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #

def scrape(settings: Settings, force: bool = False) -> list[Scheme]:
    """Scrape the listing + every detail page; cache to settings.raw_cache."""
    cache_path = Path(settings.raw_cache)
    if cache_path.exists() and not force:
        log.info("Loading cached schemes from %s", cache_path)
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Scheme(**d) for d in data]

    session = _session()
    out(f"[cyan]Fetching scheme listing:[/cyan] {settings.source_url}")
    listing_html = _get(session, settings.source_url)
    links = parse_scheme_links(listing_html, settings.source_base)
    out(f"[green]Found {len(links)} schemes[/green]")

    schemes: list[Scheme] = []
    for i, (sid, url) in enumerate(links, 1):
        out(f"  [{i}/{len(links)}] {sid} ... ")
        try:
            html = _get(session, url)
            scheme = parse_scheme_detail(html, sid, url, settings.department_id)
            schemes.append(scheme)
            log.info("Parsed scheme %s: %s", sid, scheme.title)
        except Exception as exc:
            log.error("Failed to scrape %s: %s", url, exc)
        # polite delay
        time.sleep(0.4)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([s.to_dict() for s in schemes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    out(f"[green]Cached {len(schemes)} schemes ->[/green] {cache_path}")
    return schemes


def load_cached(settings: Settings) -> list[Scheme]:
    """Load schemes from the JSON cache (must exist)."""
    cache_path = Path(settings.raw_cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached schemes at {cache_path}. Run `python -m app.scrape` first."
        )
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return [Scheme(**d) for d in data]


if __name__ == "__main__":
    from .config import get_settings

    scrape(get_settings())
