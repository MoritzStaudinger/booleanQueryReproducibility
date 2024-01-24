import json
from copy import copy

from retriv import DenseRetriever


def prepare_data(review_data):
    collection = [
        {"id": doc["document_id"], "text": doc["text"]}
        for doc in review_data["data"]["train"]
    ]
    qrels = {
        doc["document_id"]: int(doc["labels"][0])
        for doc in review_data["data"]["train"]
    }
    return collection, qrels


def prepare_seed(seed_collection):
    collection = [
        {"id": doc["id"], "text": doc["title"]}
        for doc in seed_collection
    ]
    qrels = {
        doc["id"]: 1
        for doc in seed_collection
    }
    return collection, qrels


def create_retriever(documents: list[dict[str, str]]):
    dr = DenseRetriever(
        index_name="topic-similarity",
        model="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
        max_length=128,
        use_ann=True,
    )

    dr = dr.index(
        collection=documents,
        use_gpu=False,  # Default value
        batch_size=512,  # Default value
        show_progress=True,  # Default value
        # callback=lambda doc: {  # Callback defaults to None.
        #     "id": doc["id"],
        #     "text": doc["title"] + ". " + doc["text"],
        # },
    )
    return dr


if __name__ == '__main__':
    with open('../../input/Seed/overall_collection.jsonl', 'r') as f:
        seed_collection = [json.loads(line) for line in f.readlines()]

    top_similar = {}
    for item in seed_collection:
        copied_collection = copy(seed_collection)

        copied_collection.remove(item)

        documents, qrels = prepare_seed(copied_collection)

        dr = create_retriever(documents=documents)
        top_similar[item] = dr.search(
            query=documents[0]['text'],  # What to search for
            return_docs=True,  # Default value, return the text of the documents
            cutoff=10,  # Default value, number of results to return
        )

    with open('../../data/2-runs/seed/seed.json', 'w') as f:
        json.dump(top_similar, f)
