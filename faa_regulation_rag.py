"""
FAA / CFR regulation RAG: PDF + text ingest, chunk, embed, ChromaDB, similarity search.

Designed to stay separate from the ASRS incident Chroma collection. No LLM required for retrieval.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

import requests
from pypdf import PdfReader


# Bumped when chunking / ingest logic changes (invalidates Chroma skip-cache without editing files).
INGEST_CHUNKER_KEY = "charwin-v7"

# Default OpenAI-style embeddings URL for EurON (confirm in your dashboard; path may differ).
DEFAULT_EURON_EMBEDDINGS_URL = "https://api.euron.one/api/v1/euri/embeddings"

# High-priority ACs from your capstone brief (add PDFs under corpus_dir; see FAA doc library).
PRIORITY_ADVISORY_CIRCULARS = [
    "AC 90-114A",
    "AC 90-100A",
    "AC 91-74B",
    "AC 120-51E",
    "AC 120-71B",
    "AC 20-138D",
    "AC 25-11B",
    "AC 25-7D",
    "AC 20-115D",
    "AC 00-24C",
    "AC 00-45H",
    "AC 91-51A",
    "AC 60-22",
    "AC 120-92A",
    "AC 120-103A",
]

# Official bulk XML (dated issue). The public HTML "current" pages often block Colab/scripts.
ECFR_TITLES_API = "https://www.ecfr.gov/api/versioner-import/v1/titles"
ECFR_TITLE14_XML = "https://www.ecfr.gov/api/versioner/v1/full/{issue_date}/title-14.xml"

_ECFR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AcademicResearch/1.0; "
        "regulation-text-ingest; +https://www.ecfr.gov/developers/documentation/api/v1)"
    ),
    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
}

FAA_DOC_LIBRARY_AC_BASE = "https://www.faa.gov/documentLibrary/media/Advisory_Circular"


def _ecfr_title14_issue_date(timeout: float = 60.0) -> str:
    r = requests.get(ECFR_TITLES_API, timeout=timeout, headers=_ECFR_HEADERS)
    r.raise_for_status()
    data = r.json()
    titles = data.get("titles", data) if isinstance(data, dict) else data
    for t in titles:
        if not isinstance(t, dict):
            continue
        if int(t.get("number", -1)) == 14:
            d = t.get("up_to_date_as_of") or t.get("latest_issue_date")
            if d:
                return str(d)
    raise RuntimeError("Could not resolve Title 14 issue date from eCFR titles API.")


def _extract_part_91_xml_text(xml_bytes: bytes) -> str:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)

    def local_tag(el: Any) -> str:
        tag = el.tag
        return tag.split("}")[-1] if "}" in tag else tag

    part_el = None
    for elem in root.iter():
        if local_tag(elem) == "DIV5" and elem.get("N") == "91" and elem.get("TYPE") == "PART":
            part_el = elem
            break
    if part_el is None:
        raise RuntimeError("Part 91 (DIV5 N=91 TYPE=PART) not found in title-14 XML.")

    words: list[str] = []
    for t in part_el.itertext():
        s = (t or "").strip()
        if s:
            words.append(s)
    return "\n".join(words)


def _is_bot_block_or_placeholder(text: str) -> bool:
    low = text.lower()
    needles = (
        "aggressive automated scraping",
        "programmatic access to these sites is limited",
        "complete the captcha",
        "bot test",
        "federalregister.gov api documentation",
    )
    return any(n in low for n in needles)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corpus_fingerprint(corpus_dir: str) -> str:
    """Hash of all .pdf / .txt under corpus_dir (paths + size + content hash)."""
    corpus_dir = os.path.abspath(corpus_dir)
    if not os.path.isdir(corpus_dir):
        return "empty"
    entries = []
    for root, _, files in os.walk(corpus_dir):
        for name in sorted(files):
            if not name.lower().endswith((".pdf", ".txt")):
                continue
            p = os.path.join(root, name)
            rel = os.path.relpath(p, corpus_dir)
            st = os.stat(p)
            entries.append((rel, st.st_size, _sha256_file(p)))
    h = hashlib.sha256()
    for rel, sz, digest in sorted(entries):
        h.update(f"{rel}|{sz}|{digest}\n".encode())
    return h.hexdigest()


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        parts.append(t)
    return "\n\n".join(parts)


def load_document_text(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        return extract_pdf_text(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def fetch_ecfr_part_91_to_txt(
    out_path: str,
    *,
    issue_date: str | None = None,
    timeout: float = 180.0,
) -> str:
    """
    Download Title 14 via the official eCFR versioner API, extract 14 CFR Part 91 as plain text.

    Uses ``/api/versioner-import/v1/titles`` to pick a valid ``issue_date``, then
    ``/api/versioner/v1/full/{date}/title-14.xml`` (same data the site documents for developers).
    This avoids the HTML reader pages, which often return a bot-block message for Colab/scripts.

    If you previously saved the HTML block page as ``14_cfr_part_91_ecfr.txt``, delete it and
    re-run, or point ``out_path`` at a new filename.
    """
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    date = issue_date or _ecfr_title14_issue_date(timeout=min(timeout, 120.0))
    url = ECFR_TITLE14_XML.format(issue_date=date)
    r = requests.get(url, timeout=timeout, headers=_ECFR_HEADERS)
    r.raise_for_status()
    text = _extract_part_91_xml_text(r.content)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if _is_bot_block_or_placeholder(text) or len(text) < 4000:
        raise RuntimeError(
            "eCFR Part 91 extraction failed or produced too little text. "
            "Try a specific issue_date (YYYY-MM-DD from the titles API) or add a Part 91 PDF under the corpus folder."
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- eCFR Title 14 XML issue_date={date} source_api=versioner/v1/full -->\n")
        f.write(text)
    return out_path


def _normalize_ac_number(ac: str) -> str:
    """
    Convert inputs like:
    - "AC 90-114A"
    - "90-114A"
    - "ac_90-114a"
    to a canonical "90-114A" string.
    """
    s = (ac or "").strip().upper()
    s = re.sub(r"^AC[\s_-]*", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "-")
    return s


def _candidate_ac_filenames(ac_number: str) -> list[str]:
    """
    Generate common FAA document-library naming patterns for AC PDFs.
    """
    n = _normalize_ac_number(ac_number)
    if not n:
        return []
    compact = n.replace("-", "")
    return [
        f"AC_{n}.pdf",        # AC_90-114A.pdf
        f"AC-{n}.pdf",        # AC-90-114A.pdf
        f"AC_{compact}.pdf",  # AC_90114A.pdf
        f"AC-{compact}.pdf",  # AC-90114A.pdf
    ]


def fetch_faa_advisory_circular_pdf(
    ac_number: str,
    out_dir: str,
    *,
    timeout: float = 120.0,
    overwrite: bool = False,
) -> str:
    """
    Download one Advisory Circular PDF from FAA document library into out_dir.

    Returns local file path on success. Raises RuntimeError if all known URL
    filename patterns fail.
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ac_norm = _normalize_ac_number(ac_number)
    if not ac_norm:
        raise ValueError(f"Invalid AC number: {ac_number!r}")

    local_name = f"AC_{ac_norm}.pdf"
    local_path = os.path.join(out_dir, local_name)
    if os.path.isfile(local_path) and not overwrite:
        return local_path

    errors: list[str] = []
    headers = {"User-Agent": _ECFR_HEADERS["User-Agent"]}
    for file_name in _candidate_ac_filenames(ac_norm):
        url = f"{FAA_DOC_LIBRARY_AC_BASE}/{file_name}"
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            if r.status_code != 200:
                errors.append(f"{url} -> HTTP {r.status_code}")
                continue
            ctype = (r.headers.get("Content-Type") or "").lower()
            payload = r.content or b""
            # Basic PDF sanity check to avoid writing HTML placeholders.
            if b"%PDF" not in payload[:1024] and "pdf" not in ctype:
                errors.append(f"{url} -> non-PDF response ({ctype or 'unknown content-type'})")
                continue
            with open(local_path, "wb") as f:
                f.write(payload)
            return local_path
        except Exception as ex:
            errors.append(f"{url} -> {ex}")
            continue

    detail = "\n".join(errors[-6:]) if errors else "No URL attempts were made."
    raise RuntimeError(
        f"Could not download AC {ac_norm} from FAA document library.\n"
        f"Tried common filename patterns under {FAA_DOC_LIBRARY_AC_BASE}.\n"
        f"Recent errors:\n{detail}"
    )


def fetch_priority_advisory_circulars(
    corpus_dir: str,
    *,
    advisory_list: list[str] | None = None,
    timeout: float = 120.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Download priority AC PDFs into corpus_dir.

    Returns a summary dict:
      {
        "downloaded": [<paths>],
        "skipped_existing": [<paths>],
        "failed": [{"ac": "...", "error": "..."}],
      }
    """
    out_dir = os.path.abspath(corpus_dir)
    os.makedirs(out_dir, exist_ok=True)
    acs = advisory_list or PRIORITY_ADVISORY_CIRCULARS
    downloaded: list[str] = []
    skipped_existing: list[str] = []
    failed: list[dict[str, str]] = []

    for ac in acs:
        ac_norm = _normalize_ac_number(ac)
        target = os.path.join(out_dir, f"AC_{ac_norm}.pdf")
        if os.path.isfile(target) and not overwrite:
            skipped_existing.append(target)
            continue
        try:
            p = fetch_faa_advisory_circular_pdf(
                ac,
                out_dir,
                timeout=timeout,
                overwrite=overwrite,
            )
            downloaded.append(p)
        except Exception as ex:
            failed.append({"ac": str(ac), "error": str(ex)})

    return {
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "failed": failed,
    }


def chunk_text_by_chars(
    text: str,
    max_chars: int = 900,
    overlap: int = 150,
) -> list[str]:
    """Sliding character windows (used when embeddings come from EurON API — no local tokenizer)."""
    text = (text or "").strip()
    if not text:
        return []
    max_chars = max(200, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))
    step = max(1, max_chars - overlap)
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        piece = text[i : i + max_chars].strip()
        if len(piece) >= 40:
            chunks.append(piece)
        if i + max_chars >= n:
            break
        i += step
    return chunks


def euron_openai_embeddings(
    texts: list[str],
    *,
    api_key: str,
    url: str,
    model: str,
    batch_size: int = 32,
    timeout: float = 180.0,
) -> list[list[float]]:
    """
    OpenAI-compatible ``POST .../embeddings`` using the same Bearer key as EurON chat.

    Request body: ``{"model": "<name>", "input": ["...", ...]}``.
    Response: ``{"data": [{"embedding": [...], "index": i}, ...]}``.
    """
    if not api_key or not url or not model:
        raise ValueError("euron_openai_embeddings requires api_key, url, and model.")
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    out: list[list[float]] = []
    bs = max(1, int(batch_size))
    for start in range(0, len(texts), bs):
        batch = [t if isinstance(t, str) else str(t) for t in texts[start : start + bs]]
        r = requests.post(
            url.strip(),
            headers=headers,
            json={"model": model.strip(), "input": batch},
            timeout=timeout,
        )
        if r.status_code != 200:
            hint = ""
            if r.status_code == 403:
                hint = (
                    "\n(403 often means EurON wallet / daily token limit — add funds, wait until reset, "
                    "or set EURON_EMBEDDINGS_MODEL to \"\" in the notebook to use local MiniLM for FAA RAG only.)"
                )
            raise RuntimeError(
                f"Embeddings HTTP {r.status_code}: {r.text[:800]}\n"
                f"URL={url!r}. Confirm path in Euri dashboard if 404; otherwise see message above.{hint}"
            )
        data = r.json()
        items = sorted(
            (data.get("data") or []),
            key=lambda x: int(x.get("index", 0)),
        )
        if len(items) != len(batch):
            raise RuntimeError(
                f"Embeddings response size mismatch: got {len(items)} vectors for batch of {len(batch)}"
            )
        for it in items:
            emb = it.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Invalid embedding entry in API response")
            out.append([float(x) for x in emb])
    if len(out) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: {len(out)} vs {len(texts)} texts")
    return out


def _truncate_decode(tokenizer: Any, text: str, max_length: int) -> str:
    """Truncate to ``max_length`` tokens (decode round-trip); keeps ``model.encode`` quiet."""
    if not text or max_length <= 0:
        return text
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _chunks_from_token_ids(
    tokenizer: Any, enc: list[int], max_tokens: int, overlap: int
) -> list[str]:
    if not enc:
        return []
    stride = max(1, max_tokens - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(enc):
        piece = enc[start : start + max_tokens]
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True).strip())
        start += stride
    return [c for c in chunks if len(c) >= 40]


def chunk_by_model_tokens(
    text: str,
    tokenizer: Any,
    max_tokens: int = 512,
    overlap: int = 64,
) -> list[str]:
    """
    Token-bounded chunks for the embedding model.

    Part 91 (and similar) can be 100k+ tokens. A single ``encode(full_text)`` triggers
    length warnings and can fail. We therefore scan **fixed-size character windows**
    (each well below model context), encode only each window, then split window token
    ids into ``max_tokens`` slices with overlap.
    """
    text = (text or "").strip()
    if not text:
        return []
    overlap = max(0, min(overlap, max_tokens - 1))
    # Character window before tokenization. Dense CFR can exceed ~0.55 tokens/char (see 512→279).
    # Keep ``encode`` under ``max_tokens`` via ``truncation=True`` and a modest char budget.
    char_cap = max(160, min(int(max_tokens * 1.7), 900))
    char_stride = max(1, char_cap - max(40, overlap * 2))

    all_chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        window = text[i : i + char_cap]
        enc = tokenizer.encode(
            window,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )
        all_chunks.extend(_chunks_from_token_ids(tokenizer, enc, max_tokens, overlap))
        if i + char_cap >= n:
            break
        i += char_stride
    return all_chunks


@dataclass
class IngestResult:
    n_chunks: int
    fingerprint: str
    chroma_path: str


def ingest_faa_corpus(
    corpus_dir: str,
    chroma_persist_dir: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "faa_regulations",
    max_tokens: int = 512,
    overlap: int = 64,
    force_rebuild: bool = False,
    batch_size: int = 64,
    *,
    euron_api_key: str | None = None,
    euron_embeddings_url: str | None = None,
    euron_embeddings_model: str | None = None,
    euron_char_chunk: int = 900,
    euron_char_overlap: int = 150,
    auto_fetch_priority_acs: bool = True,
    advisory_fetch_timeout: float = 120.0,
) -> IngestResult:
    """
    Read all .pdf and .txt under corpus_dir, chunk, embed, upsert into Chroma.

    **Embeddings:** default is local ``sentence-transformers`` (e.g. MiniLM). If you pass
    ``euron_api_key``, ``euron_embeddings_url``, and ``euron_embeddings_model``, uses
    OpenAI-compatible ``/embeddings`` on the EurON gateway (same key as chat — not the
    chat/completions endpoint).
    """
    import chromadb

    corpus_dir = os.path.abspath(corpus_dir)
    chroma_persist_dir = os.path.abspath(chroma_persist_dir)
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(chroma_persist_dir, exist_ok=True)

    if auto_fetch_priority_acs:
        ac_fetch = fetch_priority_advisory_circulars(
            corpus_dir,
            timeout=advisory_fetch_timeout,
            overwrite=False,
        )
        print(
            "FAA AC fetch:",
            f"downloaded={len(ac_fetch.get('downloaded', []))},",
            f"skipped_existing={len(ac_fetch.get('skipped_existing', []))},",
            f"failed={len(ac_fetch.get('failed', []))}",
        )
        failed = ac_fetch.get("failed", []) or []
        if failed:
            # Keep logs compact but actionable.
            for item in failed[:3]:
                print(f"  - AC fetch failed: {item.get('ac')} | {item.get('error')}")
            if len(failed) > 3:
                print(f"  ... and {len(failed) - 3} more failures")

    use_euron = bool(
        (euron_api_key or "").strip()
        and (euron_embeddings_url or "").strip()
        and (euron_embeddings_model or "").strip()
    )

    fp_raw = corpus_fingerprint(corpus_dir)
    embed_tag = (
        f"euron|{euron_embeddings_url}|{euron_embeddings_model}|c{euron_char_chunk}|o{euron_char_overlap}"
        if use_euron
        else f"st|{embedding_model_name}|mt{max_tokens}|o{overlap}"
    )
    fp = hashlib.sha256(f"{INGEST_CHUNKER_KEY}|{embed_tag}|{fp_raw}".encode()).hexdigest()

    tokenizer = None
    model = None
    content_cap = 0
    overlap_use = 0
    if not use_euron:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(embedding_model_name)
        tokenizer = model.tokenizer
        _seq = int(getattr(model, "max_seq_length", 256) or 256)
        cap_toks = max(32, min(int(max_tokens), _seq))
        content_cap = max(48, cap_toks - 24)
        overlap_use = max(0, min(int(overlap), content_cap - 1, content_cap // 2))

    client = chromadb.PersistentClient(path=chroma_persist_dir)
    coll = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if not force_rebuild:
        try:
            if coll.count() > 0:
                sample = coll.get(limit=1, include=["metadatas"])
                md0 = (sample["metadatas"] or [{}])[0] or {}
                if md0.get("corpus_fingerprint") == fp:
                    return IngestResult(
                        n_chunks=coll.count(),
                        fingerprint=fp,
                        chroma_path=chroma_persist_dir,
                    )
        except Exception:
            pass

    # Rebuild collection
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    coll = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for root, _, files in os.walk(corpus_dir):
        for name in sorted(files):
            if not name.lower().endswith((".pdf", ".txt")):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, corpus_dir)
            doc_type = "CFR" if "part-91" in name.lower() or "part_91" in name.lower() or "14_cfr" in name.lower() else "OTHER"
            if "ac_" in name.lower() or "advisory" in name.lower():
                doc_type = "AC"
            if "safo" in name.lower():
                doc_type = "SAFO"

            raw = load_document_text(path)
            if len(raw.strip()) < 30:
                continue
            if use_euron:
                chunks = chunk_text_by_chars(
                    raw,
                    max_chars=int(euron_char_chunk),
                    overlap=int(euron_char_overlap),
                )
            else:
                assert tokenizer is not None
                chunks = chunk_by_model_tokens(
                    raw, tokenizer, max_tokens=content_cap, overlap=overlap_use
                )
            citation = os.path.splitext(name)[0].replace("_", " ")
            for i, ch in enumerate(chunks):
                cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{fp}:{rel}:{i}:{ch[:80]}"))
                ids.append(cid)
                documents.append(ch)
                metadatas.append(
                    {
                        "corpus_fingerprint": fp,
                        "source_file": rel,
                        "citation": citation,
                        "doc_type": doc_type,
                        "chunk_index": i,
                    }
                )

    if not ids:
        raise FileNotFoundError(
            f"No ingestible text found under {corpus_dir}. "
            f"Add PDFs/TXT (see PRIORITY_ADVISORY_CIRCULARS) or run fetch_ecfr_part_91_to_txt()."
        )

    _ids, _docs, _mds = [], [], []
    for cid, doc, md in zip(ids, documents, metadatas):
        if use_euron:
            clipped = (doc or "").strip()[:24000]
        else:
            assert tokenizer is not None
            clipped = _truncate_decode(tokenizer, doc, content_cap)
        if len(clipped) >= 40:
            _ids.append(cid)
            _docs.append(clipped)
            _mds.append(md)
    ids, documents, metadatas = _ids, _docs, _mds

    if not ids:
        raise FileNotFoundError("All regulation chunks were empty after truncation; check corpus files.")

    if use_euron:
        vec_rows: list[list[float]] = euron_openai_embeddings(
            documents,
            api_key=euron_api_key or "",
            url=(euron_embeddings_url or "").strip(),
            model=(euron_embeddings_model or "").strip(),
            batch_size=min(batch_size, 32),
        )
    else:
        assert model is not None
        _saved_msl = getattr(model, "max_seq_length", None)
        model.max_seq_length = int(content_cap)
        try:
            _emb = model.encode(
                documents, show_progress_bar=True, batch_size=batch_size
            )
        finally:
            if _saved_msl is not None:
                model.max_seq_length = _saved_msl
        vec_rows = _emb.tolist()

    for start in range(0, len(ids), 500):
        sl = slice(start, start + 500)
        coll.upsert(
            ids=ids[sl],
            embeddings=vec_rows[sl],
            documents=documents[sl],
            metadatas=metadatas[sl],
        )

    return IngestResult(n_chunks=len(ids), fingerprint=fp, chroma_path=chroma_persist_dir)


def query_trend_for_regulations(
    trend_text: str,
    chroma_persist_dir: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "faa_regulations",
    top_k: int = 5,
    *,
    euron_api_key: str | None = None,
    euron_embeddings_url: str | None = None,
    euron_embeddings_model: str | None = None,
    euron_query_max_chars: int = 12000,
) -> list[dict[str, Any]]:
    """Return top_k chunks with cosine distance → relevance score in (0,1]."""
    import chromadb

    use_euron = bool(
        (euron_api_key or "").strip()
        and (euron_embeddings_url or "").strip()
        and (euron_embeddings_model or "").strip()
    )

    client = chromadb.PersistentClient(path=os.path.abspath(chroma_persist_dir))
    coll = client.get_collection(collection_name)

    if use_euron:
        q_text = (trend_text or "").strip()[: max(500, int(euron_query_max_chars))]
        q = euron_openai_embeddings(
            [q_text],
            api_key=euron_api_key or "",
            url=(euron_embeddings_url or "").strip(),
            model=(euron_embeddings_model or "").strip(),
            batch_size=1,
        )[0]
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(embedding_model_name)
        tokenizer = model.tokenizer
        _seq = int(getattr(model, "max_seq_length", 256) or 256)
        q_cap = max(48, _seq - 24)
        q_text = _truncate_decode(tokenizer, trend_text, q_cap)
        _saved_msl = getattr(model, "max_seq_length", None)
        model.max_seq_length = int(q_cap)
        try:
            q = model.encode([q_text], show_progress_bar=False)[0].tolist()
        finally:
            if _saved_msl is not None:
                model.max_seq_length = _saved_msl
    res = coll.query(
        query_embeddings=[q],
        n_results=min(top_k, max(1, coll.count())),
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict[str, Any]] = []
    for i in range(len(res["ids"][0])):
        dist = (res["distances"] or [[0.0]])[0][i]
        # Chroma cosine space: distance = 1 - cosine_similarity for normalized vectors
        relevance = max(0.0, min(1.0, 1.0 - float(dist)))
        hits.append(
            {
                "citation": (res["metadatas"] or [[{}]])[0][i].get("citation", ""),
                "doc_type": (res["metadatas"] or [[{}]])[0][i].get("doc_type", ""),
                "source_file": (res["metadatas"] or [[{}]])[0][i].get("source_file", ""),
                "text": (res["documents"] or [[""]])[0][i],
                "relevance": round(relevance, 4),
            }
        )
    return hits


def format_regulation_report(trend: str, hits: Iterable[dict[str, Any]]) -> str:
    lines = [
        f"TREND: {trend.strip()}",
        "",
        "RELEVANT REGULATIONS:",
        "━" * 40,
    ]
    for h in hits:
        icon = {"AC": "📋", "SAFO": "⚠", "CFR": "§"}.get(h.get("doc_type", ""), "📄")
        cite = h.get("citation") or h.get("source_file", "unknown")
        rel = h.get("relevance", 0.0)
        excerpt = (h.get("text") or "").strip()
        if len(excerpt) > 450:
            excerpt = excerpt[:447] + "..."
        lines.append(f"{icon} {cite}  (relevance: {rel:.2f})")
        lines.append(f'   "{excerpt}"')
        lines.append("")
    return "\n".join(lines).rstrip()
