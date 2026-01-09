import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    # You can add other config variables here later (e.g., UPLOAD_FOLDER)