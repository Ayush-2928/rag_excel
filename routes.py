from flask import Blueprint, request, render_template, jsonify
from better_profanity import profanity
import validators
from rag_system import RAGSystem

# Create a Blueprint named 'main'
main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/set_name', methods=['POST'])
def set_name():
    user_name = request.form.get('name')
    if not user_name:
        return jsonify({'error': "Name is required."}), 400
    return jsonify({'message': "Name set successfully.", 'name': user_name})

@main.route('/process_documents', methods=['POST'])
def process_documents():
    user_name = request.form.get('user_name')
    rag_system = RAGSystem(user_name)
    input_type = request.form.get('input_type')

    texts = []
    success = False

    if input_type == "PDFs":
        pdf_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_pdfs(pdf_files)
    elif input_type == "Word Files":
        docx_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_word(docx_files)
    elif input_type == "Excel Files":
        excel_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_excel(excel_files)
    elif input_type == "TXT Files":
        txt_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_txt(txt_files)
    elif input_type == "URLs":
        urls = request.form.get("urls").splitlines()
        url_list = [url.strip() for url in urls if url.strip()]
        if not all(validators.url(url) for url in url_list):
            return jsonify({'error': "Please enter valid URLs."}), 400
        texts, success = rag_system.fetch_url_content(url_list)
    else:
        return jsonify({'error': "Invalid input type."}), 400

    if not success:
        return jsonify({'error': texts}), 400

    rag_system.vectorize_content(texts)
    return jsonify({'message': "Documents processed and embeddings generated successfully."})

@main.route('/answer_question', methods=['POST'])
def answer_question():
    user_name = request.form.get('user_name')
    query = request.form.get('query')

    if profanity.contains_profanity(query):
        return jsonify({'error': "Please use appropriate language."}), 400
    if not query:
        return jsonify({'error': "Please provide a query."}), 400

    rag_system = RAGSystem(user_name)
    try:
        rag_system.load_embeddings()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    
    relevant_chunks = rag_system.retrieve_relevant_content(query)
    combined_context = ' '.join(relevant_chunks)
    answer = rag_system.generate_answer(combined_context, query)

    return jsonify({
        'relevant_chunks': relevant_chunks,
        'answer': answer
    })

@main.route('/chat', methods=['POST'])
def chat():
    user_name = request.form.get('user_name')
    user_message = request.form.get('message')

    if profanity.contains_profanity(user_message):
        return jsonify({'response': "Please use appropriate language."}), 400
    if not user_message:
        return jsonify({'error': "Please enter a message."}), 400

    rag_system = RAGSystem(user_name)
    response = rag_system.chat_with_llm(user_message)
    return jsonify({'response': response})