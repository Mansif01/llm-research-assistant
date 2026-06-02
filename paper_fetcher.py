import requests
import chromadb
from sentence_transformers import SentenceTransformer
import time
import xml.etree.ElementTree as ET

# ============================================================
# SETUP -- same embedding model and database as ingest.py
# ============================================================

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./paper_database")
collection = chroma_client.get_or_create_collection(
    name="research_papers",
    metadata={"hnsw:space": "cosine"}
)


# ============================================================
# SEMANTIC SCHOLAR -- search by topic, gets abstracts
# ============================================================

def search_semantic_scholar(query, max_papers=15):
    """
    Search Semantic Scholar for papers matching a topic.
    Returns abstracts -- no PDF download needed.
    Free API, no key required.

    Example:
        search_semantic_scholar("data quality visualization", 15)
    """

    print(f"\nSearching Semantic Scholar: {query}")

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": max_papers,
        "fields": "title,authors,year,abstract,venue,externalIds"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
    except requests.exceptions.Timeout:
        print("  Semantic Scholar timed out -- skipping")
        return 0

    if response.status_code != 200:
        print(f"  API error: {response.status_code}")
        return 0

    papers = response.json().get("data", [])
    added_count = 0

    for paper in papers:

        # Skip papers with no abstract
        if not paper.get("abstract"):
            continue

        title = paper.get("title", "Unknown Title")
        year = paper.get("year", "Unknown")
        venue = paper.get("venue", "Unknown venue")

        # Build author string
        all_authors = paper.get("authors", [])
        authors = ", ".join([
            a.get("name", "")
            for a in all_authors[:3]
        ])
        if len(all_authors) > 3:
            authors += " et al."

        abstract = paper.get("abstract", "")

        # Build the full text to store and search
        full_text = f"""Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
Source: Semantic Scholar

Abstract:
{abstract}"""

        # Convert to embedding vector
        embedding = embedding_model.encode([full_text]).tolist()

        # Create a unique ID for this paper
        safe_title = title[:40].replace(' ', '_').replace('/', '_')
        paper_id = f"ss_{safe_title}_{year}"

        try:
            collection.add(
                documents=[full_text],
                embeddings=embedding,
                ids=[paper_id],
                metadatas=[{
                    "title": title,
                    "authors": authors,
                    "year": str(year),
                    "venue": venue,
                    "source": "Semantic Scholar",
                    "chunk_index": 0
                }]
            )
            added_count += 1
            print(f"  + Added: {title[:55]}...")

        except Exception:
            # Already in database -- skip silently
            pass

        # Small delay to be polite to the API
        time.sleep(2)

    print(f"  Done: {added_count} papers added")
    return added_count


# ============================================================
# ARXIV -- search preprints, good for latest AI/CS papers
# ============================================================

def search_arxiv(query, max_papers=10):
    """
    Search ArXiv for CS and AI preprints.
    ArXiv has the latest research before it appears elsewhere.
    Free, no API key needed.

    Example:
        search_arxiv("retrieval augmented generation", 10)
    """

    print(f"\nSearching ArXiv: {query}")

    url = "http://export.arxiv.org/api/query"

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_papers,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
    except requests.exceptions.Timeout:
        print("  ArXiv timed out -- skipping")
        return 0

    if response.status_code != 200:
        print(f"  ArXiv error: {response.status_code}")
        return 0

    # ArXiv returns XML -- parse it
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        print("  Could not parse ArXiv response")
        return 0

    entries = root.findall('atom:entry', namespace)
    added_count = 0

    for entry in entries:

        title_el = entry.find('atom:title', namespace)
        summary_el = entry.find('atom:summary', namespace)
        published_el = entry.find('atom:published', namespace)

        if title_el is None or summary_el is None:
            continue

        title = title_el.text.strip().replace('\n', ' ')
        abstract = summary_el.text.strip().replace('\n', ' ')
        year = published_el.text[:4] if published_el is not None else "Unknown"

        # Get authors
        all_authors = entry.findall('atom:author', namespace)
        author_names = []
        for author in all_authors:
            name_el = author.find('atom:name', namespace)
            if name_el is not None:
                author_names.append(name_el.text)

        authors = ", ".join(author_names[:3])
        if len(author_names) > 3:
            authors += " et al."

        full_text = f"""Title: {title}
Authors: {authors}
Year: {year}
Source: ArXiv

Abstract:
{abstract}"""

        embedding = embedding_model.encode([full_text]).tolist()

        safe_title = title[:40].replace(' ', '_').replace('/', '_')
        paper_id = f"arxiv_{safe_title}_{year}"

        try:
            collection.add(
                documents=[full_text],
                embeddings=embedding,
                ids=[paper_id],
                metadatas=[{
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "source": "ArXiv",
                    "chunk_index": 0
                }]
            )
            added_count += 1
            print(f"  + Added: {title[:55]}...")

        except Exception:
            pass

        time.sleep(1)

    print(f"  Done: {added_count} papers added")
    return added_count


# ============================================================
# COMBINED SEARCH -- searches BOTH sources at once
# ============================================================

def search_all_sources(query, ss_papers=10, arxiv_papers=8):
    """
    Search both Semantic Scholar AND ArXiv for a query.
    This gives maximum coverage.

    Example:
        search_all_sources("interactive data visualization", 10, 8)
    """

    print(f"\n{'=' * 50}")
    print(f"Searching all sources for: {query}")
    print(f"{'=' * 50}")

    ss_count = search_semantic_scholar(query, ss_papers)
    arxiv_count = search_arxiv(query, arxiv_papers)

    total = ss_count + arxiv_count
    print(f"\nTotal added for '{query}': {total} papers")
    return total


# ============================================================
# BULK POPULATE -- add papers for many topics at once
# ============================================================

def bulk_populate(topics, ss_per_topic=10, arxiv_per_topic=5):
    """
    Add papers for a list of topics automatically.
    Use this to build your initial database.

    Example:
        bulk_populate([
            "data quality visualization",
            "retrieval augmented generation",
            "human computer interaction"
        ])
    """

    grand_total = 0

    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] Topic: {topic}")
        count = search_all_sources(topic, ss_per_topic, arxiv_per_topic)
        grand_total += count

        # Wait between topics to avoid rate limiting
        if i < len(topics):
            print("  Waiting 10 seconds before next topic...")
            time.sleep(10)

    print(f"\n{'=' * 50}")
    print(f"BULK POPULATE COMPLETE")
    print(f"Total papers added: {grand_total}")
    print(f"Database now contains: {get_database_stats()['total']} entries")
    print(f"{'=' * 50}")

    return grand_total


# ============================================================
# DATABASE STATS -- see what is in your database
# ============================================================

def get_database_stats():
    """Return stats about the current database"""
    total = collection.count()

    stats = {"total": total}

    # Get sample of papers
    if total > 0:
        sample = collection.peek(min(10, total))
        papers = []
        seen_titles = set()

        for metadata in sample['metadatas']:
            title = metadata.get('title', 'Unknown')
            if title not in seen_titles:
                seen_titles.add(title)
                papers.append({
                    "title": title,
                    "year": metadata.get('year', '?'),
                    "source": metadata.get('source', 'Unknown')
                })

        stats["sample_papers"] = papers

    return stats


# ============================================================
# RUN THIS FILE DIRECTLY TO POPULATE YOUR DATABASE
# ============================================================

if __name__ == "__main__":
    # These topics are chosen to match your dissertation
    # background and the LLM research assistant project
    topics = [
        "data profiling visualization interactive",
        "data quality assessment cleaning tools",
        "retrieval augmented generation LLM",
        "large language models research assistant",
        "human computer interaction data exploration",
        "information visualization techniques D3",
        "natural language processing knowledge retrieval",
        "interactive machine learning user interface"
    ]

    bulk_populate(topics, ss_per_topic=10, arxiv_per_topic=8)