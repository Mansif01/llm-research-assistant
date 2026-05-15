import anthropic
import os
from ingest import search_papers, embedding_model
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def generate_research_answer(user_question, conversation_history=None):
    """
    Main RAG function:
    1. Search database for relevant paper chunks
    2. Build a prompt with those chunks as context
    3. Ask Claude to answer using that context
    4. Return answer with citations
    """

    # Step 1 -- retrieve relevant chunks
    search_results = search_papers(user_question, n_results=5)

    # Step 2 -- format retrieved context
    context_pieces = []
    sources = []

    for i, (doc, metadata) in enumerate(zip(
            search_results['documents'][0],
            search_results['metadatas'][0]
    )):
        context_pieces.append(
            f"[Source {i + 1}] From '{metadata['title']}' "
            f"({metadata['authors']}, {metadata['year']}):\n{doc}"
        )

        # Track unique sources for citations
        source_key = metadata['title']
        if source_key not in [s['title'] for s in sources]:
            sources.append({
                'title': metadata['title'],
                'authors': metadata['authors'],
                'year': metadata['year']
            })

    context = "\n\n".join(context_pieces)

    # Step 3 -- build the prompt
    system_prompt = """You are a helpful research assistant with access to 
    a database of academic papers. Your job is to help researchers by:

    1. Answering questions about research topics accurately
    2. Suggesting research ideas and hypotheses
    3. Recommending relevant papers and explaining why they are relevant
    4. Explaining complex concepts clearly

    IMPORTANT RULES:
    - Only use information from the provided context to answer questions
    - Always cite which source you are drawing from using [Source N]
    - If the context does not contain enough information say so clearly
    - Suggest follow-up questions the researcher might find useful
    - Keep answers concise but complete"""

    # Build conversation messages
    messages = []

    # Add conversation history if provided
    if conversation_history:
        messages.extend(conversation_history)

    # Add current question with context
    messages.append({
        "role": "user",
        "content": f"""Based on the following research paper excerpts, 
        please answer my question.

CONTEXT FROM PAPERS:
{context}

MY QUESTION: {user_question}

Please answer based on the context above and cite your sources."""
    })

    # Step 4 -- call Claude
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        system=system_prompt,
        messages=messages
    )

    answer = response.content[0].text

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(context_pieces)
    }


def generate_research_ideas(topic, existing_work=""):
    """
    Generate novel research ideas based on a topic
    Uses retrieved papers to ground the suggestions
    """

    # Search for relevant work
    search_results = search_papers(
        f"research gaps limitations future work {topic}",
        n_results=8
    )

    context_pieces = []
    for doc, metadata in zip(
            search_results['documents'][0],
            search_results['metadatas'][0]
    ):
        context_pieces.append(
            f"From '{metadata['title']}':\n{doc}"
        )

    context = "\n\n".join(context_pieces)

    prompt = f"""Based on these research paper excerpts about {topic}, 
    generate 5 novel research ideas or hypotheses.

CONTEXT:
{context}

{f"EXISTING WORK BY RESEARCHER: {existing_work}" if existing_work else ""}

For each idea provide:
1. A clear research question
2. Why it is novel (what gap it addresses)
3. A suggested methodology
4. Expected impact
5. Which papers from the context support this direction

Focus on ideas that are achievable and have clear academic contribution."""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def recommend_papers(research_interest, n_papers=5):
    """
    Recommend papers from the database relevant to a research interest
    with explanations of why each is relevant
    """

    search_results = search_papers(research_interest, n_results=n_papers * 2)

    # Deduplicate by paper title
    seen_titles = set()
    unique_papers = []

    for doc, metadata in zip(
            search_results['documents'][0],
            search_results['metadatas'][0]
    ):
        if metadata['title'] not in seen_titles:
            seen_titles.add(metadata['title'])
            unique_papers.append({
                'title': metadata['title'],
                'authors': metadata['authors'],
                'year': metadata['year'],
                'excerpt': doc[:300]
            })

    # Ask Claude to explain relevance
    papers_text = "\n\n".join([
        f"Paper: {p['title']} ({p['authors']}, {p['year']})\n"
        f"Excerpt: {p['excerpt']}"
        for p in unique_papers[:n_papers]
    ])

    prompt = f"""A researcher is interested in: {research_interest}

Here are papers from our database:
{papers_text}

For each paper explain in 2-3 sentences why it is relevant to 
the researcher's interest. Be specific about what they will learn."""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "papers": unique_papers[:n_papers],
        "explanations": response.content[0].text
    }