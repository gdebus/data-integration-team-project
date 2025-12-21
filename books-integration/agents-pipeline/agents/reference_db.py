import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


DB_DIR = "./pydi_reference_db"


def load_text_file(txt_path: str) -> str:
    """
    Read all text content from reference file.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"TXT file not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def build_reference_db(txt_path: str):
    """
    Build and persist a Chroma vector DB from the reference.
    """
    print(f"Loading text file: {txt_path}")
    text = load_text_file(txt_path)

    print("Splitting into chunks...")
    chunks = split_text(text)

    print(f"Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Building vector DB...")
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    db.persist()

    print(f"Vector DB built and saved at: {DB_DIR}")
    return db


def load_reference_db():
    """
    Load the previously built vector DB.
    """
    if not os.path.exists(DB_DIR):
        raise RuntimeError("Vector DB not found. Run build_reference_db first.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )


def query_pydi_reference(query: str) -> str:
    """
    Search the PyDI reference text document for relevant content.
    Returns top-4 extracted chunks.
    """
    db = load_reference_db()
    results = db.similarity_search(query, k=4)

    if not results:
        return "No relevant reference content found."

    output = "\n\n-----\n\n".join([r.page_content for r in results])
    return output


if __name__ == "__main__":

    ROOT = Path.cwd()
    INPUT_DIR = ROOT / "books-integration" / "agents-pipeline" / "agents" / "input"

    build_reference_db(str(INPUT_DIR / "reference_pydi.txt"))
    result = query_pydi_reference("How do I perform entity matching in PyDI?")
    print(result)
