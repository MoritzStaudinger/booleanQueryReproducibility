import json
from copy import copy
import os
from retriv import DenseRetriever


def prepare_seed(seed_collection):
    collection = [
        {"id": doc["id"], "text": doc["title"]}
        for doc in seed_collection
    ]
    return collection


def create_retriever(documents: list[dict[str, str]]):
    dr = DenseRetriever(
        index_name="topic-similarity",
        model="pritamdeka/S-PubMedBert-MS-MARCO",
        normalize=True,
        max_length=258,
        use_ann=True,
    )

    dr = dr.index(
        collection=documents,
        use_gpu=False,  # Default value
        batch_size=512,  # Default value
        show_progress=True,  # Default value
    )
    return dr


if __name__ == '__main__':
    with open('../../input/Seed/overall_collection.jsonl', 'r') as f:
        seed_collection = [json.loads(line) for line in f.readlines()]
    seed_collection = prepare_seed(seed_collection)

    tar_collection = []
    for filename in os.listdir('../../input/CLEF_TAR/2018'):
        with open(f'../../input/CLEF_TAR/2018/{filename}', 'r') as f:
            title_stub = f.readlines()[2]

            tar_collection.append({
                'id': filename,
                'text': title_stub[7:]
            })

    combined_collection = seed_collection + tar_collection

    top_similar = {}
    for item in combined_collection:
        copied_collection = copy(combined_collection)

        # remove the item from the collection
        copied_collection.remove(item)

        # documents = prepare_seed(copied_collection)

        dr = create_retriever(documents=copied_collection)

        results = dr.search(
            query=item['text'],  # What to search for
            return_docs=False,  # Default value, return the text of the documents
            cutoff=30,  # Default value, number of results to return
        )
        top_similar[item['id']] = {_id: float(score) for _id, score in results.items()}

    with open('../../data/0-qrels/seed-topic-similarity.json', 'w') as f:
        json.dump(top_similar, f)

