import json
import logging
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INDEX_DIR = "data/metric_index"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s"
)

logger = logging.getLogger(__name__)


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def load_index() -> object:

    index_path = Path(
        INDEX_DIR
    ) / "index.faiss"

    index = faiss.read_index(
        str(index_path)
    )

    return index


def load_metadata() -> list:

    metadata_path = Path(
        INDEX_DIR
    ) / "metadata.json"

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    return metadata


def search_metric(
    query: str,
    index: object,
    metadata: list
) -> None:

    query_embedding = model.encode(
        [query]
    )

    distances, indexes = index.search(
        query_embedding,
        3
    )

    for i in range(len(indexes[0])):

        index_number = indexes[0][i]

        if index_number == -1:
            continue

        metric = metadata[index_number]

        logger.info(
            "Metric: %s",
            metric["metric_name"]
        )

        logger.info(
            "Criteria: %s",
            metric["criteria"]
        )

        logger.info(
            "Distance: %.4f",
            distances[0][i]
        )

        print()


def main() -> None:

    index = load_index()

    metadata = load_metadata()

    query = input(
        "Enter your query: "
    )

    search_metric(
        query,
        index,
        metadata
    )


if __name__ == "__main__":
    main()