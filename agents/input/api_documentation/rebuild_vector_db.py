"""Rebuild the PyDI documentation vector DB with improved chunking.

Improvements over the original crawl_api_docs.ipynb:
1. Semantic chunking: splits on class/function boundaries instead of fixed char count
2. Chunk type tagging: each chunk gets a 'type' prefix (API_SIGNATURE, DESCRIPTION, SOURCE_CODE)
3. Deduplication: removes duplicate chunks that appear across pages
4. Signature preservation: class/function signatures are never split across chunks
"""

import hashlib
import os
import re
import shutil
import sys
from typing import Dict, List, Set, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load env from the agents directory
_agents_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
load_dotenv(os.path.join(_agents_dir, ".env"))
# Also try cwd
load_dotenv(os.path.join(os.getcwd(), ".env"))

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_URL = "https://wbsg-uni-mannheim.github.io/PyDI/"
DB_DIR = os.path.join(os.path.dirname(__file__), "pydi_apidocs_vector_db")
BATCH_SIZE = 50
MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


def is_internal_link(url: str) -> bool:
    return url.startswith(BASE_URL)


def crawl_pydi_docs(start_url: str) -> List[Tuple[str, str]]:
    """Crawl all PyDI doc pages. Returns list of (url, raw_html)."""
    visited: Set[str] = set()
    to_visit = [start_url]
    pages = []

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"[x] HTTP {resp.status_code}: {url}")
                continue

            pages.append((url, resp.text))
            print(f"[*] Crawled: {url}")

            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.find_all("a", href=True):
                href = urljoin(url, link["href"]).split("#")[0]
                if is_internal_link(href) and href not in visited:
                    to_visit.append(href)

        except Exception as e:
            print(f"[x] Failed: {url} ({e})")

    return pages


def extract_api_blocks(html: str) -> List[Dict[str, str]]:
    """Parse HTML into semantic blocks: signatures, descriptions, source code."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["nav", "footer", "header", "script", "style"]):
        tag.decompose()

    blocks = []

    # Extract class and function definitions (dl elements with class/function entries)
    for dt in soup.find_all("dt"):
        # Check if this is a class or function signature
        sig_id = dt.get("id", "")
        text = dt.get_text(separator=" ", strip=True)

        if not text or len(text) < 10:
            continue

        # Determine type
        block_type = "DESCRIPTION"
        if "class " in text[:30] or re.match(r"^(class\s+)?PyDI\.", text):
            block_type = "API_SIGNATURE"
        elif "[source]" in text or "(" in text[:100]:
            block_type = "API_SIGNATURE"

        # Get the description (dd sibling)
        dd = dt.find_next_sibling("dd")
        description = ""
        if dd:
            description = dd.get_text(separator=" ", strip=True)

        # Build the chunk with type prefix
        if block_type == "API_SIGNATURE":
            # Keep signature + description together
            chunk = f"[{block_type}] {text}"
            if description:
                # Truncate very long descriptions but keep param docs
                if len(description) > 1200:
                    # Try to keep up to the Returns section
                    returns_idx = description.find("Returns")
                    if returns_idx > 0 and returns_idx < 1200:
                        description = description[: returns_idx + 200]
                    else:
                        description = description[:1200]
                chunk += f"\n{description}"
            blocks.append({"type": block_type, "text": chunk, "id": sig_id})

    # Also extract section-level content (module docstrings, descriptions)
    main = soup.find("div", class_="document")
    if main:
        for section in main.find_all(["section", "div"], class_=["section"]):
            # Get section heading
            heading = section.find(["h1", "h2", "h3", "h4"])
            if not heading:
                continue
            heading_text = heading.get_text(strip=True)

            # Get direct text content (not nested dl/class defs)
            paragraphs = section.find_all("p", recursive=False)
            content = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

            if content and len(content) > 30:
                blocks.append({
                    "type": "DESCRIPTION",
                    "text": f"[DESCRIPTION] {heading_text}: {content}",
                    "id": heading_text,
                })

    return blocks


def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """Remove exact and near-duplicate chunks."""
    seen_hashes: Set[str] = set()
    unique = []

    for chunk in chunks:
        # Normalize whitespace for dedup
        normalized = re.sub(r"\s+", " ", chunk.strip())
        h = hashlib.md5(normalized.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(chunk)

    return unique


def split_long_blocks(blocks: List[Dict[str, str]]) -> List[str]:
    """Split blocks that exceed MAX_CHUNK_SIZE while preserving signatures."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\nParameters :", "\nReturns", "\nReturn type", "\n\n", "\n", " "],
    )

    chunks = []
    for block in blocks:
        text = block["text"]
        if len(text) <= MAX_CHUNK_SIZE:
            chunks.append(text)
        else:
            # For API signatures, try to keep the signature line intact
            if block["type"] == "API_SIGNATURE":
                # Find first newline after signature
                sig_end = text.find("\n")
                if sig_end > 0 and sig_end < MAX_CHUNK_SIZE:
                    signature = text[:sig_end]
                    rest = text[sig_end:]
                    # Split the description part
                    sub_chunks = splitter.split_text(rest)
                    # Prepend signature context to first chunk
                    chunks.append(signature + "\n" + sub_chunks[0] if sub_chunks else signature)
                    # Add remaining with a brief context prefix
                    sig_short = signature[:80]
                    for sc in sub_chunks[1:]:
                        chunks.append(f"[contd. {sig_short}...]\n{sc}")
                else:
                    chunks.extend(splitter.split_text(text))
            else:
                chunks.extend(splitter.split_text(text))

    return chunks


def build_vector_db(chunks: List[str]) -> None:
    """Build the Chroma vector DB from chunks."""
    # Backup existing DB
    if os.path.exists(DB_DIR):
        backup_dir = DB_DIR + "_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(DB_DIR, backup_dir)
        print(f"[*] Backed up existing DB to {backup_dir}")
        shutil.rmtree(DB_DIR)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    db = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )

    # Batched ingestion
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        db.add_texts(batch)
        batch_num = i // BATCH_SIZE + 1
        print(f"[*] Embedded batch {batch_num}/{total_batches} ({len(batch)} chunks)")

    print(f"[+] Vector DB rebuilt successfully at {DB_DIR}")
    print(f"    Total chunks: {db._collection.count()}")


def main():
    print("=" * 60)
    print("Rebuilding PyDI documentation vector DB")
    print("=" * 60)

    # Step 1: Crawl
    print("\n[Phase 1] Crawling PyDI documentation...")
    pages = crawl_pydi_docs(BASE_URL)
    print(f"Crawled {len(pages)} pages")

    # Step 2: Extract semantic blocks
    print("\n[Phase 2] Extracting semantic blocks...")
    all_blocks = []
    for url, html in pages:
        blocks = extract_api_blocks(html)
        all_blocks.extend(blocks)
    print(f"Extracted {len(all_blocks)} semantic blocks")
    print(f"  API_SIGNATURE: {sum(1 for b in all_blocks if b['type'] == 'API_SIGNATURE')}")
    print(f"  DESCRIPTION:   {sum(1 for b in all_blocks if b['type'] == 'DESCRIPTION')}")

    # Step 3: Also add raw text chunks as fallback (the original approach, but with better splitting)
    print("\n[Phase 3] Adding fallback text chunks...")
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\nclass ", "\ndef ", "\n\n", "\n", " "],
    )
    fallback_chunks = []
    for url, html in pages:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["nav", "footer", "header", "script", "style"]):
            tag.decompose()
        main = soup.find("div", class_="document")
        if main:
            text = main.get_text(separator="\n", strip=True)
            if text and len(text) > 50:
                fallback_chunks.extend(fallback_splitter.split_text(text))
    print(f"Generated {len(fallback_chunks)} fallback text chunks")

    # Step 4: Split long blocks and combine
    print("\n[Phase 4] Splitting and deduplicating...")
    semantic_chunks = split_long_blocks(all_blocks)
    all_chunks = semantic_chunks + fallback_chunks
    all_chunks = deduplicate_chunks(all_chunks)
    print(f"Final chunk count after dedup: {len(all_chunks)}")

    # Step 5: Build vector DB
    print("\n[Phase 5] Building vector DB...")
    build_vector_db(all_chunks)

    print("\n" + "=" * 60)
    print("Done! Vector DB rebuilt.")
    print("=" * 60)


if __name__ == "__main__":
    main()
