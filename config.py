import os 
from dotenv import load_dotenv 

load_dotenv() 

WHISPER_MODEL =os.getenv("WHISPER_MODEL")

RECORDINGS_DIR = os.getenv("RECORDINGS_DIR")
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR")

OLLAMA_BASE_URL =os.getenv("OLLAMA_BASE_URL")
EVALUATOR_MODEL=os.getenv("EVALUATOR_MODEL")
