import json 
import logging 
from pathlib import Path
import faiss 
from sentence_transformers import SentenceTransformer

logger=logging.getLogger(__name__) 
logging.basicConfig(level=logging.INFO , format="%(levelname)s : %(message)s")

chunk_size =500 
index_directory ="data/transcript_index"

def load_model():

    model =SentenceTransformer("all-MiniLM-L6-v2")
    return model 

def load_transcript(file_path:str )->str :
    with open(file_path , "r" , encoding="utf-8")as file:
        transcript=file.read() 
    return transcript 

def create_chunks(transcript:str)->list:
    chunks=[] 
    start =0 
    while start< len(transcript):
        chunk =transcript[start:start+chunk_size]
        chunks.append(chunk) 
        start =start +chunk_size 
    return chunks

def create_embeddings(model , chunks: list):
    embeddings=model.encode(chunks) 
    return embeddings 

def create_index(embeddings):
    dimension=embeddings.shape[1] 
    index=faiss.IndexFlatL2(dimension) 
    index.add(embeddings)
    return index 

def save_index(index , chunks:list , transcript_path:str):
    recording_name =Path(transcript_path).stem

    index_dir =(Path(index_directory)/ recording_name)
    index_dir.mkdir(parents=True , exist_ok=True)
    index_file= index_dir/"index.faiss" 
    metadata_file =index_dir/"metadata.json"

    faiss.write_index(index , str(index_file))

    metadata=[] 

    for i , chunk in enumerate(chunks):
        metadata.append({
            "chunk_id":i , 
            "text":chunk
        })

    with open(metadata_file , "w" , encoding="utf-8")as file:
        json.dump(metadata , file , indent=4) 

    logger.info("faiss index saved to : %s" , index_file)
    logger.info("metadata saved to: %s" , metadata_file) 

def build_transcript_index(transcript_path:str):
    transcript =load_transcript(transcript_path) 
    chunks =create_chunks(transcript) 
    logger.info("created %d chunks" , len(chunks))

    model =load_model() 
    embeddings=create_embeddings(model , chunks)
    index =create_index(embeddings) 
    save_index(index , chunks , transcript_path) 

    logger.info("transcript index created successfully") 


if __name__ =="__main__":
    transcript_path =input("enter transcript path : ") 
    # transcript_path ="transcripts/recording_3.txt" 

    if not Path(transcript_path).exists():
        logger.error("transcript file not found: %s" ,transcript_path) 
    else:
        build_transcript_index(transcript_path)     
                