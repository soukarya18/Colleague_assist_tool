import json 
import logging 
from pathlib import Path
import faiss 
from sentence_transformers import SentenceTransformer
from openai import OpenAI 

from config import OLLAMA_BASE_URL , EVALUATOR_MODEL

logger =logging.getLogger(__name__) 
logging.basicConfig(level=logging.INFO , format="%(levelname)s : %(message)s")

metrics_file ="data/metrics.json" 
index_directory="data/transcript_index" 
results_directory ="results" 

def load_metrics()->list:
    logger.info("loading metrics....") 
    with open(metrics_file , "r" , encoding="utf-8")as file:
        data=json.load(file) 
    return data["metrics"] 

def load_transcript_index(transcript_path:str)->tuple:

    recording_name=Path(transcript_path).stem
    index_directory=Path("data/transcript_index")/recording_name

    index_file =index_directory/"index.faiss" 
    metadata_file =index_directory/"metadata.json" 
    logger.info("loading transcript index for : %s" , recording_name) 

    index =faiss.read_index(str(index_file))

    with open(metadata_file , "r" ,encoding="utf-8")as file:
        metadata=json.load(file) 
    return index,metadata 

def load_embedding_model():
    logger.info("loading embedding model...")
    model =SentenceTransformer( "all-MiniLM-L6-v2")
    return model


def search_chunks(model , index , metadata , criteria)->list:
    logger.info("searching transcript for criteria: %s" , criteria)
    query_embedding=model.encode([criteria]) 
    distances,indexes =index.search(query_embedding,2) 
    retrieved_chunks=[] 

    for distance,chunk_index in zip(distances[0],indexes[0]):
        chunk=metadata[chunk_index] 
        retrieved_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "distance": float(distance),
                "text": chunk["text"]
            }
        )
    return retrieved_chunks   


def create_prompt(metric:dict , retrieved_chunks:list)->str:
    criteria=metric["criteria"][0] 
    prompt =f"""

You are a customer support call quality auditor.

Your task is to evaluate whether the customer support agent
satisfied the given metric.

Metric Name:
{metric["name"]}

Criteria:
{criteria}

Metric Type:
{metric["metric_type"]}

Below are the relevant parts of the call transcript.

"""

    for chunk in retrieved_chunks:
        prompt+=f"""
        Transcript chunk :
        {chunk["text"]}
        """
    prompt+= """
Evaluate the metric using only the transcript evidence provided.

Rules:

1. PASS if the transcript contains direct evidence that the agent asked
   for the information described by the criteria.

2. Treat semantically equivalent wording as satisfying the criteria.
   For example:
   - "May I have your name?" satisfies a criterion asking for the
     customer's first/last/full name.
   - "Do you have an email address?" satisfies a criterion asking
     whether the agent asked for email.

3. Do not require the agent to use the exact wording from the criteria.

4. FAIL only when the transcript shows that the agent did not ask for
   the required information.

5. NOT_APPLICABLE should be used only when the metric genuinely cannot
   be evaluated from the provided evidence.

6. Prefer direct transcript evidence over assumptions or interpretation.

7. The agent's question is sufficient evidence of satisfying an
   "asked by agent" metric; the customer's response is additional
   supporting evidence.

Return the response only in valid JSON format:

{
    "result": "PASS",
    "justification": "Short explanation of why the metric passed or failed",
    "evidence": "Relevant statement from the transcript"
}
"""
    return prompt 


def evaluate_metric(
    client: object,
    metric: dict,
    retrieved_chunks: list
) -> dict:

    logger.info(
        "Evaluating metric: %s",
        metric["name"]
    )

    prompt = create_prompt(
        metric,
        retrieved_chunks
    )

    response = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    response_text = response.choices[0].message.content

    result = json.loads(
        response_text
    )

    return result


def calculate_score(
    metric: dict,
    evaluation: dict
) -> float:

    if evaluation["result"] == "PASS":

        return metric["weight"]

    return 0.0


def save_results(
    results: dict,
    recording_name: str
) -> str:

    results_path = Path(results_directory)

    results_path.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        results_path /
        f"{recording_name}_analysis.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    logger.info(
        "Results saved to: %s",
        output_file
    )

    return str(output_file)


def evaluate_transcript(
    transcript_path: str
) -> str:

    if not Path(transcript_path).exists():

        logger.error(
            "Transcript file not found: %s",
            transcript_path
        )

        return ""

    recording_name = Path(
        transcript_path
    ).stem

    metrics = load_metrics()

    index, metadata = load_transcript_index(
        transcript_path
    )

    model = load_embedding_model()

    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama"
    )

    metric_results = []

    total_score = 0.0

    maximum_score = 0.0
    fatal_metric_failed=False 

    for metric in metrics:

        logger.info(
            "Processing metric: %s",
            metric["name"]
        )

        criteria = metric["criteria"][0]

        retrieved_chunks = search_chunks(
            model,
            index,
            metadata,
            criteria
        )

        evaluation = evaluate_metric(
            client,
            metric,
            retrieved_chunks
        )

        score = calculate_score(
            metric,
            evaluation
        )

        
        if metric["metric_type"].lower()=="fatal" and evaluation["result"].lower()=="fail":
            fatal_metric_failed=True 

        total_score += score

        maximum_score += metric["weight"]

        metric_results.append(
            {
                "name": metric["name"],
                "metric_type": metric["metric_type"],
                "weight": metric["weight"],
                "result": evaluation["result"],
                "score": score,
                "justification": evaluation["justification"],
                "evidence": evaluation["evidence"],
                "retrieved_chunks": retrieved_chunks
            }
        )

    if fatal_metric_failed:
        total_score=0    

    else:
        total_score= (total_score/maximum_score )*100     

    results = {
        "recording": recording_name,
        "metrics": metric_results,
        "total_score": total_score,
        "maximum_score": maximum_score
    }

    result_path = save_results(
        results,
        recording_name
    )

    return result_path


def main() -> None:

    transcript_path = input(
        "Enter transcript path: "
    ).strip()

    result_path = evaluate_transcript(
        transcript_path
    )

    if result_path:

        print(
            f"\nAnalysis saved to: {result_path}"
        )

if __name__ == "__main__":
    main()    
            
