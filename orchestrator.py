"""
main.py

1. Show existing metrics
2. Ask if user wants to add/delete/update metrics
3. Get call recording path
4. Transcribe recording
5. Get generated transcript path
6. Build FAISS index for that transcript
7. Evaluate all metrics
8. Save final JSON result
9. Show result path

"""

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    filename="orchestrator.log",
    filemode ="w" ,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True 
)

from pathlib import Path

from transcriber import transcribe_audio, save_transcript
from transcript_index import build_transcript_index
from evaluator import evaluate_transcript
from metric_manager import load_metrics,modify_metrics, show_metrics , save_metrics




def manage_metrics():
    metrics =load_metrics() 
    print("\nPreviously selected metrics:\n")

    show_metrics(metrics)

    choice = input(
        "\nDo you want to modify the metrics? (y/n): "
    ).strip().lower()

    if choice == "y":
        metrics =load_metrics()
        changed = modify_metrics(metrics)
        if changed:
            save_metrics(metrics) 
        
        print("\nUpdated metrics:\n")
        show_metrics(metrics)

    else:

        print(
            "\nUsing existing metrics."
        )

def process_call(audio_path:str):
    logger.info("starting call processing....")
    logger.info("step 1: transcribing call recording...")

    segments =transcribe_audio(audio_path) 
    transcript_path =save_transcript(segments , audio_path) 

    logger.info("transcript generated : %s" , transcript_path)
    logger.info("creating transcript index....")

    build_transcript_index(transcript_path) 

    logger.info("evaluating call.....")

    result_path=evaluate_transcript(transcript_path) 

    logger.info("call evaluation completed...")
    print(
        "\n--------------------------------"
    )

    print(
        "Call processing completed successfully."
    )

    print(
        f"Analysis saved to: {result_path}"
    )

    print(
        "--------------------------------"
    )


def main():
    print(
        "\n================================"
    )

    print(
        "CALL AUDIT ASSIST TOOL"
    )

    print(
        "================================"
    )

    manage_metrics() 
    audio_path= input("\nEnter call recording path: ").strip() 
    if not Path(audio_path).exists():
        logger.error("audio file not found: %s" , audio_path) 
        return 
    process_call(audio_path) 



if __name__ =="__main__":
    main() 