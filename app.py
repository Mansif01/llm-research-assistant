from flask import Flask, render_template, request, jsonify, session
from rag_engine import (
    generate_research_answer,
    generate_research_ideas,
    recommend_papers
)
from ingest import add_paper_to_database
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.route('/')
def index():
    # Give each user a session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['conversation_history'] = []
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint -- handles general research questions"""
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Get conversation history from session
    history = session.get('conversation_history', [])

    # Generate response
    result = generate_research_answer(user_message, history)

    # Update conversation history
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": result['answer']})

    # Keep only last 10 exchanges to avoid token limits
    if len(history) > 20:
        history = history[-20:]

    session['conversation_history'] = history

    return jsonify({
        'answer': result['answer'],
        'sources': result['sources'],
        'retrieved_chunks': result['retrieved_chunks']
    })


@app.route('/ideas', methods=['POST'])
def get_ideas():
    """Generate research ideas for a given topic"""
    data = request.json
    topic = data.get('topic', '')
    existing_work = data.get('existing_work', '')

    if not topic:
        return jsonify({'error': 'No topic provided'}), 400

    ideas = generate_research_ideas(topic, existing_work)

    return jsonify({'ideas': ideas})


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Recommend papers for a research interest"""
    data = request.json
    interest = data.get('interest', '')

    if not interest:
        return jsonify({'error': 'No interest provided'}), 400

    result = recommend_papers(interest)

    return jsonify(result)


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    session['conversation_history'] = []
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    app.run(debug=True)