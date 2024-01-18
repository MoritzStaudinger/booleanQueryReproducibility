import pandas as pd
from Bio import Entrez


def temporal_submission(
    query: str, email: str, mindate: str, maxdate: str
) -> tuple[int, list[str]]:
    """
    Return the number of results and the list of IDs for a given query and date range.

    :param query:
    :param email:
    :param mindate:
    :param maxdate:
    :return:
    """
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=10000,
        email=email,
        mindate=mindate.replace("-", "/"),
        maxdate=maxdate.replace("-", "/"),
    )
    record = Entrez.read(handle)

    return int(record["Count"]), record["IdList"]


if __name__ == "__main__":
    email = "test@test.com"
    date = ("1946/01/01", "2018/12/31")

    df = pd.read_csv("../../output/Seed_gpt-3.5-turbo-1106_3007195.csv")

    queries = {
        "query": "query",
        "edited_search": "edited_search",
        "q1_answer": "q1_answer",
        "q2_answer": "q2_answer",
        "q3_answer": "q3_answer",
        "q4_answer": "q4_answer",
        "q5_answer": "q5_answer",
    }

    for index, row in df.iterrows():
        for query in queries:
            if pd.isna(row[query]):
                continue
            count, id_list = temporal_submission(
                query=row[query], email=email, mindate=date[0], maxdate=date[1]
            )
            print(count)
            print()
