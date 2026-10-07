import os 
from dotenv import load_dotenv 
from langchain_groq import ChatGroq
from langchain_nvidia import ChatNVIDIA

load_dotenv() 
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
llm = ChatGroq(
    model=EVALUATOR_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0
)

# llm = ChatNVIDIA(
#     model=EVALUATOR_MODEL,
#     api_key=NVIDIA_API_KEY,
#     temperature=0
# )
WHISPER_MODEL =os.getenv("WHISPER_MODEL")

RECORDINGS_DIR = os.getenv("RECORDINGS_DIR")
TRANSCRIPTS_DIR = os.getenv("TRANSCRIPTS_DIR")

# OLLAMA_BASE_URL =os.getenv("OLLAMA_BASE_URL")

