import os
from huggingface_hub import InferenceClient
from ingest import search_papers
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# HUGGING FACE CLIENT SETUP
# Uses Llama 3 8B -- free, no billing required
# ============================================================

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=HF_TOKEN
)


# ============================================================
# HELPER -- safe way to call the API with error handling
# ============================================================

def call_llm(messages, max_tokens=1000, temperature=0.3):
    """
    Central function that calls Hugging Face.
    All three main functions use this so errors
    are handled in one place.
    """
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)

        # Give helpful error messages for common problems
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return "ERROR: Your Hugging Face token is invalid. Check your .env file."

        if "503" in error_msg or "loading" in error_msg.lower():
            return ("The model is loading on Hugging Face servers. "
                    "This happens after a period of inactivity. "
                    "Please wait 30 seconds and try again.")

        if "429" in error_msg or "rate" in error_msg.lower():
            return ("Rate limit reached. "
                    "Hugging Face free tier allows limited requests per hour. "
                    "Please wait a few minutes and try again.")

        print(f"Hugging Face error: {e}")
        return f"Something went wrong: {error_msg}"


# ============================================================
# GENERATE RESEARCH ANSWER
# ============================================================

def generate_research_answer(user_question, conversation_history=None):
    """
    Main RAG function.
    1. Searches your paper database for relevant content
    2. Sends that content plus the question to Llama 3
    3. Returns an answer with source citations
    """

    # Step 1 -- search the database
    search_results = search_papers(user_question, n_results=5)

    # Step 2 -- check if database has papers
    if (not search_results['documents'] or
            len(search_results['documents'][0]) == 0):
        return {
            "answer": ("Your paper database is empty. "
                       "Use the 'Add Papers to Database' panel in the "
                       "sidebar to search for and add papers first. "
                       "Try searching for 'data quality visualization' "
                       "or 'retrieval augmented generation'."),
            "sources": [],
            "retrieved_chunks": 0
        }

    # Step 3 -- build context from retrieved paper chunks
    context_pieces = []
    sources = []
    seen_titles = set()

    for i, (doc, metadata) in enumerate(zip(
        search_results['documents'][0],
        search_results['metadatas'][0]
    )):
        title = metadata.get('title', 'Unknown Title')
        context_pieces.append(
            f"[Source {i + 1}] From '{title}':\n{doc}"
        )

        # Collect unique papers for citation display
        if title not in seen_titles:
            seen_titles.add(title)
            sources.append({
                "title": title,
                "authors": metadata.get('authors', 'Unknown'),
                "year": metadata.get('year', 'N/A')
            })

    context = "\n\n".join(context_pieces)

    # Step 4 -- build the messages for Llama 3
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful academic research assistant. "
                "Answer questions using ONLY the provided context from papers. "
                "Always cite your sources using [Source N]. "
                "If the context does not contain enough information, "
                "say so clearly rather than making things up. "
                "Be concise but thorough."
            )
        },
        {
            "role": "user",
            "content": (
                f"QUESTION: {user_question}\n\n"
                f"CONTEXT FROM PAPERS:\n{context}\n\n"
                f"Please answer the question based on the context above "
                f"and cite your sources."
            )
        }
    ]

    # Step 5 -- call Llama 3
    answer = call_llm(messages, max_tokens=1000, temperature=0.3)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(context_pieces)
    }


# ============================================================
# GENERATE RESEARCH IDEAS
# ============================================================

def generate_research_ideas(topic, existing_work=""):
    """
    Generate novel research ideas based on a topic.
    Searches for limitations and gaps in existing papers
    then uses Llama 3 to suggest new research directions.
    """

    # Search for papers mentioning gaps and future work
    search_results = search_papers(
        f"limitations future work gaps {topic}",
        n_results=6
    )

    if (not search_results['documents'] or
            len(search_results['documents'][0]) == 0):
        return (
            "No papers found in database for this topic. "
            "Please add relevant papers first using the sidebar."
        )

    # Build context from retrieved papers
    context = "\n\n".join([
        f"From '{m.get('title', 'Unknown')}':\n{d}"
        for d, m in zip(
            search_results['documents'][0],
            search_results['metadatas'][0]
        )
    ])

    # Add existing work section if provided
    existing_section = ""
    if existing_work.strip():
        existing_section = (
            f"\nRESEARCHER'S EXISTING WORK:\n{existing_work}\n"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert academic advisor helping a "
                "Master's student identify novel research opportunities. "
                "Be specific, practical, and grounded in the literature."
            )
        },
        {
            "role": "user",
            "content": (
                f"Based on the research papers below, generate 5 novel "
                f"research ideas about: {topic}\n"
                f"{existing_section}\n"
                f"For each idea provide:\n"
                f"1. A clear research question\n"
                f"2. Why it is novel (what gap it fills)\n"
                f"3. Suggested methodology\n"
                f"4. Expected impact\n\n"
                f"CONTEXT FROM PAPERS:\n{context}"
            )
        }
    ]

    return call_llm(messages, max_tokens=1200, temperature=0.7)


# ============================================================
# RECOMMEND PAPERS
# ============================================================

def recommend_papers(interest, n_papers=5):
    """
    Recommend papers from the database relevant to
    a research interest with explanations of relevance.
    """

    # Search for relevant papers
    search_results = search_papers(interest, n_results=n_papers * 2)

    if (not search_results['documents'] or
            len(search_results['documents'][0]) == 0):
        return {
            "papers": [],
            "explanations": (
                "No matching papers found. "
                "Try adding more papers using the sidebar first."
            )
        }

    # Deduplicate papers by title
    unique_papers = []
    seen = set()

    for doc, metadata in zip(
        search_results['documents'][0],
        search_results['metadatas'][0]
    ):
        title = metadata.get('title', 'Unknown')
        if title not in seen:
            seen.add(title)
            unique_papers.append({
                'title': title,
                'authors': metadata.get('authors', 'Unknown'),
                'year': metadata.get('year', 'N/A'),
                'excerpt': doc[:200]
            })

    # Build list for the prompt
    papers_text = "\n".join([
        f"- {p['title']} ({p['authors']}, {p['year']})"
        for p in unique_papers[:n_papers]
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research librarian helping a student "
                "understand why specific papers are relevant to their work. "
                "Be specific and mention concrete connections."
            )
        },
        {
            "role": "user",
            "content": (
                f"A researcher is interested in: {interest}\n\n"
                f"Here are papers from our database:\n{papers_text}\n\n"
                f"For each paper write exactly 2 sentences explaining "
                f"specifically why it is relevant to this researcher's interest."
            )
        }
    ]

    explanations = call_llm(messages, max_tokens=800, temperature=0.3)

    return {
        "papers": unique_papers[:n_papers],
        "explanations": explanations
    }