import os
import sys 
import logging
from pathlib import Path

from faster_whisper import WhisperModel
from config import WHISPER_MODEL , TRANSCRIPTS_DIR

model =WhisperModel(WHISPER_MODEL , device="cpu" , compute_type="int8")
logger =logging.getLogger(__name__) 
logging.basicConfig(level=logging.INFO , format ="%(levelname)s : %(message)s")
def transcribe_audio(audio_path : str)-> object:
    logger.info("transcribing: %s" , audio_path)
    segments , info = model.transcribe(audio_path)
    logger.info("detected language: %s" , info.language)

    return segments 

def save_transcript(segments: object , audio_path:str)->str:
    Path(TRANSCRIPTS_DIR).mkdir(exist_ok=True)
    file_name=Path(audio_path).stem
    output_file=Path(TRANSCRIPTS_DIR)/ f"{file_name}.txt"
    with open(output_file , "w" , encoding="utf-8")as file:
        for segment in segments:
            file.write(
                f"[{segment.start:.2f}s --> "
                f"{segment.end:.2f}s]  "
            )
            file.write(
                f"{segment.text.strip()}\n"
            )
    print(f"transcript saved to : {output_file}") 
    logger.info("transcript saved to : %s" , output_file)

    return str(output_file)       



def main():
    Path(TRANSCRIPTS_DIR).mkdir(exist_ok=True)

    audio_path=sys.argv[1]
    if not Path(audio_path).exists():
        print("audio file not found")
        return 
    segments =transcribe_audio(audio_path)
    save_transcript(segments , audio_path)


if __name__ =="__main__":
    main()     
