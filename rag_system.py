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
import pandas as pd
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi  # NEW: Import for Keyword Search
from config import Config

class RAGSystem:
    def __init__(self, user_name, model_name='all-MiniLM-L6-v2', llm_model='llama-3.1-8b-instant'):
        self.user_name = user_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.bm25 = None  # NEW: Placeholder for BM25 index
        self.memory = ConversationBufferWindowMemory(k=5)
        self.groq_chat = ChatGroq(groq_api_key=Config.GROQ_API_KEY, model_name=llm_model)
        self.conversation = ConversationChain(llm=self.groq_chat, memory=self.memory)
        self.embeddings_file = f'embeddings_{self.user_name}.pkl'

    def save_embeddings(self, embeddings, sentences, bm25):
        # UPDATED: Now saves embeddings, sentences, AND the BM25 index
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump((embeddings, sentences, bm25), f)

    def load_embeddings(self):
        if not os.path.exists(self.embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        with open(self.embeddings_file, 'rb') as f:
            # UPDATED: Loads all three components
            data = pickle.load(f)
            # Handle backward compatibility if old file has only 2 items
            if len(data) == 3:
                embeddings, sentences, bm25 = data
                self.bm25 = bm25
            else:
                embeddings, sentences = data
                self.bm25 = None # Will need re-processing if old file
                
        self.index = create_faiss_index(embeddings)
        return sentences

    # --- File Extraction Methods (No changes here) ---
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

    def extract_text_from_excel(self, excel_files):
        texts = []
        try:
            for excel_file in excel_files:
                file_text = ""
                xls = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
                for sheet_name, df in xls.items():
                    df = df.fillna('')
                    for _, row in df.iterrows():
                        row_parts = [f"{col}: {val}" for col, val in row.items() if str(val).strip() != '']
                        if row_parts:
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
        
        # 1. Create Vector Embeddings
        embeddings = self.model.encode(sentences)
        self.index = create_faiss_index(embeddings)
        
        # 2. Create BM25 Index (Keyword Search)
        # Tokenize sentences simply by splitting on spaces for BM25
        tokenized_corpus = [doc.split(" ") for doc in sentences]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Save everything
        self.save_embeddings(embeddings, sentences, self.bm25)

    # UPDATED: Hybrid Retrieval Logic
    def retrieve_relevant_content(self, query, k=15):
        if self.index is None:
            raise ValueError("Embeddings have not been loaded yet.")
        
        sentences = self.load_embeddings()
        
        # --- Method A: Vector Search (Semantic) ---
        query_embedding = self.model.encode([query])
        D, I = self.index.search(query_embedding, k=k)
        vector_indices = I[0]
        
        # --- Method B: BM25 Search (Keyword) ---
        # If BM25 exists (it should), use it
        bm25_indices = []
        if self.bm25:
            tokenized_query = query.split(" ")
            # Get top n documents based on keyword match
            # We fetch slightly more (k) to ensure we get good overlap
            bm25_scores = self.bm25.get_scores(tokenized_query)
            # Get indices of top k scores
            bm25_indices = np.argsort(bm25_scores)[::-1][:k]

        # --- Hybrid Merge ---
        # Combine unique indices from both methods
        # This gives us the best of both worlds: conceptual match + exact keyword match
        combined_indices = list(set(vector_indices) | set(bm25_indices))
        
        # Retrieve the actual sentences
        relevant_sentences = [sentences[i] for i in combined_indices if i < len(sentences)]
        
        return relevant_sentences

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