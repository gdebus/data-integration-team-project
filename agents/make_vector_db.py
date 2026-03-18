import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Set, List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


BASE_URL = "https://wbsg-uni-mannheim.github.io/PyDI/"
DB_DIR = "./pydi_apidocs_vector_db"
BATCH_SIZE = 50


def is_internal_link(url: str) -> bool:
    return url.startswith(BASE_URL)


def clean_text(soup: BeautifulSoup) -> str:
    # remove navigation, footer, sidebar
    for tag in soup(["nav", "footer", "header", "script", "style"]):
        tag.decompose()

    main = soup.find("div", class_="document")
    if not main:
        return ""

    return main.get_text(separator="\n", strip=True)


def crawl_pydi_docs(start_url: str) -> List[str]:
    visited: Set[str] = set()
    to_visit = [start_url]
    documents = []

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue

        print(f"[*] Crawling: {url}")
        visited.add(url)

        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            text = clean_text(soup)
            if text:
                documents.append(text)

            for link in soup.find_all("a", href=True):
                href = urljoin(url, link["href"])
                href = href.split("#")[0]  # remove anchors

                if is_internal_link(href) and href not in visited:
                    to_visit.append(href)

        except Exception as e:
            print(f"[x] Failed: {url} ({e})")

    return documents


def make_db():
    docs = crawl_pydi_docs(BASE_URL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_text(doc))

    print(f"Total chunks: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Create empty Chroma DB
    db = Chroma(embedding_function=embeddings, persist_directory=DB_DIR)

    # Batched ingestion
    next_progress_chunk = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        db.add_texts(batch)
        embedded_chunks = i + len(batch)
        while embedded_chunks >= next_progress_chunk:
            print(f"Embedded chunk {next_progress_chunk}")
            next_progress_chunk += 100

    db.persist()
    print("[+] PyDI documentation vector DB built successfully")
