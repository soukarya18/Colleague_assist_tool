import json
import logging
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


index_directory = "data/transcript_index"


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s"
)


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def load_index() -> object:

    index_file = (
        Path(index_directory)
        / "index.faiss"
    )

    index = faiss.read_index(
        str(index_file)
    )

    return index


def load_metadata() -> list:

    metadata_file = (
        Path(index_directory)
        / "metadata.json"
    )

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return metadata


def search_transcript(
    query: str,
    index: object,
    metadata: list
) -> None:

    query_embedding = model.encode(
        [query]
    )

    distances, indexes = index.search(
        query_embedding,
        2
    )

    for i in range(len(indexes[0])):

        chunk_id = indexes[0][i]

        if chunk_id == -1:
            continue

        chunk = metadata[chunk_id]

        logger.info(
            "Chunk ID: %s",
            chunk["chunk_id"]
        )

        logger.info(
            "Distance: %.4f",
            distances[0][i]
        )

        print(
            f"\n{chunk['text']}"
        )

        print(
            "-" * 60
        )


def main() -> None:

    index = load_index()

    metadata = load_metadata()

    query = input(
        "Enter your query: "
    )

    search_transcript(
        query,
        index,
        metadata
    )


if __name__ == "__main__":

    main()