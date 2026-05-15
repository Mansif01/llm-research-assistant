import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Initialise the embedding model
# This converts text into vectors that can be searched
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialise ChromaDB
# This is your local vector database
chroma_client = chromadb.PersistentClient(path="./paper_database")
collection = chroma_client.get_or_create_collection(
    name="research_papers",
    metadata={"hnsw:space": "cosine"}
)


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks
    Overlap means the end of one chunk appears at the
    start of the next -- this prevents losing context
    at chunk boundaries
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks


def add_paper_to_database(pdf_path, paper_title, paper_authors, paper_year):
    """Process a paper and add it to the vector database"""

    print(f"Processing: {paper_title}")

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Split into chunks
    chunks = chunk_text(text)

    # Create embeddings for each chunk
    embeddings = embedding_model.encode(chunks).tolist()

    # Add to ChromaDB
    # Each chunk gets a unique ID and metadata
    ids = [f"{paper_title}_{i}" for i in range(len(chunks))]

    metadatas = [{
        "title": paper_title,
        "authors": paper_authors,
        "year": str(paper_year),
        "chunk_index": i
    } for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    print(f"Added {len(chunks)} chunks from {paper_title}")


def search_papers(query, n_results=5):
    """Search the database for chunks relevant to a query"""

    # Convert query to embedding
    query_embedding = embedding_model.encode([query]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )

    return results


if __name__ == "__main__":
    # Test by adding the RAG survey paper
    # Download the PDF from arxiv first
    add_paper_to_database(
        pdf_path="papers/rag_survey.pdf",
        paper_title="RAG Survey 2024",
        paper_authors="Gao et al.",
        paper_year=2024
    )

    # Test search
    results = search_papers("what is retrieval augmented generation")
    print(results['documents'][0][0][:200])