import json 
import logging 
from pathlib import Path 
import faiss
from sentence_transformers import SentenceTransformer

logger =logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO , format="%(levelname)s : %(message)s")

metrics_file ="data/metrics.json"
index_directory = "data/metric_index"

model =SentenceTransformer("all-MiniLM-L6-v2")

def load_metrics(file_path:str)->dict:
    with open(file_path , "r" , encoding="utf-8") as file:
        metrics =json.load(file)
    return metrics

def get_criteria(metrics:dict)->list:
    criteria_list=[] 
    for metric in metrics["metrics"]:
        for criteria in metric["criteria"]:
            criteria_list.append({
                "metric_name":metric["name"] , 
                "criteria":criteria
            })    
    return criteria_list

def create_embeddings(criteria_list:list) ->object:
    texts=[] 
    for item in criteria_list:
        texts.append(item["criteria"])
    embeddings=model.encode(texts)
    return embeddings

def create_index(embeddings:object)->object:
    dimension=embeddings.shape[1] 
    index=faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index 

def save_index(index:object , criteria_list:list):
    index_dir =Path(index_directory)

    index_dir.mkdir(parents=True , exist_ok=True)
    index_file =index_dir/"index.faiss" 
    metadata_file =index_dir/"metadata.json"

    faiss.write_index(index , str(index_file))

    with open(metadata_file , "w" , encoding="utf-8")as file:
        json.dump(criteria_list , file , indent=4)
    logger.info("FAISS index saved to : %s" , index_file)
    logger.info("metadata saved to: %s" , metadata_file)

def build_metric_index():
    logger.info("loading metrics.....")
    metrics=load_metrics(metrics_file)
    criteria_list=get_criteria(metrics)

    if not criteria_list:
        logger.error("No criteria found in metrics.json")
        return 
    logger.info("creating embeddings.......")

    embeddings=create_embeddings(criteria_list)
    logger.info("Creating faiss index....")

    index =create_index(embeddings)
    save_index(index , criteria_list)

    logger.info("metrics index created successfully ......") 

if __name__ =="__main__":
    build_metric_index()                