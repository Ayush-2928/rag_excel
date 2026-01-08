from flask import Flask, request, render_template, jsonify
import pypdfium2 as pdfium
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import validators
import docx2txt
import os
import pickle
import pandas as pd  # Required for Excel processing
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from flask_cors import CORS
from better_profanity import profanity

load_dotenv()

app = Flask(__name__)
CORS(app)

groq_api_key = os.environ['GROQ_API_KEY']

class RAGSystem:
    def __init__(self, user_name, model_name='all-MiniLM-L6-v2', llm_model='llama-3.1-8b-instant'):
        self.user_name = user_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.memory = ConversationBufferWindowMemory(k=5)
        self.groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        self.conversation = ConversationChain(llm=self.groq_chat, memory=self.memory)
        self.embeddings_file = f'embeddings_{self.user_name}.pkl'

    def save_embeddings(self, embeddings, sentences):
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump((embeddings, sentences), f)

    def load_embeddings(self):
        if not os.path.exists(self.embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        with open(self.embeddings_file, 'rb') as f:
            embeddings, sentences = pickle.load(f)
        self.index = create_faiss_index(embeddings)
        return sentences

    def extract_text_from_pdfs(self, pdf_files):
        texts = []
        try:
            for pdf_file in pdf_files:
                text = ""
                pdf = pdfium.PdfDocument(pdf_file)
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    textpage = page.get_textpage()
                    text += textpage.get_text_range()
                texts.append(text)
        except Exception as e:
            return str(e), False
        return texts, True

    def extract_text_from_word(self, docx_files):
        texts = []
        try:
            for docx_file in docx_files:
                text = docx2txt.process(docx_file)
                texts.append(text)
        except Exception as e:
            return str(e), False
        return texts, True

    def extract_text_from_txt(self, txt_files):
        texts = []
        try:
            for txt_file in txt_files:
                text = txt_file.read().decode("utf-8")
                texts.append(text)
        except Exception as e:
            return str(e), False
        return texts, True

    # Method to handle Excel files (multiple sheets)
    def extract_text_from_excel(self, excel_files):
        texts = []
        try:
            for excel_file in excel_files:
                file_text = ""
                # Read all sheets into a dictionary
                xls = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
                
                for sheet_name, df in xls.items():
                    # Clean data: replace NaN with empty string
                    df = df.fillna('')
                    
                    # Iterate through rows to create context-rich sentences
                    for _, row in df.iterrows():
                        # Create "Column: Value" strings
                        row_parts = [f"{col}: {val}" for col, val in row.items() if str(val).strip() != '']
                        
                        if row_parts:
                            # Construct a sentence: "In sheet 'Sales', data entry is: Date: 2024-01-01, Amount: 500."
                            row_str = f"In sheet '{sheet_name}', data entry is: " + ", ".join(row_parts) + ". "
                            file_text += row_str
                            
                texts.append(file_text)
        except Exception as e:
            return str(e), False
        return texts, True

    def fetch_url_content(self, urls):
        contents = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                text = ' '.join([p.get_text() for p in soup.find_all('p')])
                if len(text.split()) < 50:
                    text = ' '.join([div.get_text() for div in soup.find_all('div') if len(div.get_text().split()) > 50])
                contents.append(text)
            except requests.exceptions.RequestException as e:
                return str(e), False
        return contents, True

    def vectorize_content(self, texts):
        sentences = []
        for text in texts:
            sentences.extend(text.split('. '))
        
        embeddings = self.model.encode(sentences)
        self.index = create_faiss_index(embeddings)
        
        # Save embeddings and sentences to a user-specific pickle file
        self.save_embeddings(embeddings, sentences)

    # UPDATED: k=15 to fix multi-hop retrieval issues
    def retrieve_relevant_content(self, query, k=15):
        if self.index is None:
            raise ValueError("Embeddings have not been loaded yet.")
        
        query_embedding = self.model.encode([query])
        D, I = self.index.search(query_embedding, k=k)
        sentences = self.load_embeddings()
        relevant_sentences = [sentences[i] for i in I[0]]
        return relevant_sentences

    # UPDATED: Includes Chain of Thought and Markdown formatting instructions
    def generate_answer(self, context, query):
        user_message = f"""
        You are a helpful data assistant. Use the following context to answer the user's question.
        
        CONTEXT:
        {context}
        
        USER QUESTION: 
        {query}
        
        INSTRUCTIONS:
        1. If the question requires calculation or comparison (like 'highest', 'lowest', 'total'), you MUST list the relevant data points extracted from the context first.
        2. Perform the comparison or calculation step-by-step.
        3. State the final answer clearly based on your analysis.
        4. Do not make up data that is not in the context.
        5. FORMAT YOUR OUTPUT USING MARKDOWN:
           - Use **bold** for the final answer and key numbers.
           - Use bullet points for the data extraction steps.
           - Use clear headings if answering multiple parts of a question.
        """
        response = self.conversation(user_message)
        return response['response']

    def chat_with_llm(self, user_message):
        response = self.conversation(user_message)
        return response['response']

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_name', methods=['POST'])
def set_name():
    user_name = request.form.get('name')
    if not user_name:
        return jsonify({'error': "Name is required."}), 400
    return jsonify({'message': "Name set successfully.", 'name': user_name})

@app.route('/process_documents', methods=['POST'])
def process_documents():
    user_name = request.form.get('user_name')
    rag_system = RAGSystem(user_name)
    input_type = request.form.get('input_type')

    if input_type == "PDFs":
        pdf_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_pdfs(pdf_files)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "Word Files":
        docx_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_word(docx_files)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "TXT Files":
        txt_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_txt(txt_files)
        if not success:
            return jsonify({'error': texts}), 400

    # Added logic for Excel Files
    elif input_type == "Excel Files":
        excel_files = request.files.getlist("files")
        texts, success = rag_system.extract_text_from_excel(excel_files)
        if not success:
            return jsonify({'error': texts}), 400

    elif input_type == "URLs":
        urls = request.form.get("urls").splitlines()
        url_list = [url.strip() for url in urls if url.strip()]
        if not all(validators.url(url) for url in url_list):
            return jsonify({'error': "Please enter valid URLs."}), 400

        texts, success = rag_system.fetch_url_content(url_list)
        if not success:
            return jsonify({'error': texts}), 400

    else:
        return jsonify({'error': "Invalid input type."}), 400

    rag_system.vectorize_content(texts)
    return jsonify({'message': "Documents processed and embeddings generated successfully."})

@app.route('/answer_question', methods=['POST'])
def answer_question():
    user_name = request.form.get('user_name')
    query = request.form.get('query')

    if profanity.contains_profanity(query):
        return jsonify({'error': "Please use appropriate language to ask your question."}), 400
    if not query:
        return jsonify({'error': "Please provide a query."}), 400

    rag_system = RAGSystem(user_name)
    try:
        rag_system.load_embeddings()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    
    # Retrieves 15 chunks (defined in class default)
    relevant_chunks = rag_system.retrieve_relevant_content(query)
    combined_context = ' '.join(relevant_chunks)
    answer = rag_system.generate_answer(combined_context, query)

    return jsonify({
        'relevant_chunks': relevant_chunks,
        'answer': answer
    })

@app.route('/chat', methods=['POST'])
def chat():
    user_name = request.form.get('user_name')
    user_message = request.form.get('message')

    if profanity.contains_profanity(user_message):
        return jsonify({'response': "Please use appropriate language to chat."}), 400
    
    if not user_message:
        return jsonify({'error': "Please enter a message to chat with the LLM."}), 400

    rag_system = RAGSystem(user_name)
    response = rag_system.chat_with_llm(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=8001)