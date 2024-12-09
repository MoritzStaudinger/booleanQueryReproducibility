import argparse
import os
import urllib
from http.client import IncompleteRead
from urllib.error import HTTPErro

import pandas as pd
from Bio import Entrez
from ranx import Run
from tqdm import tqdm


def temporal_submission(
    query: str, email: str, mindate: str, maxdate: str
) -> tuple[int, list[str]]:
    """
    Return the number of results and the list of IDs for a given query and date range.

    :param query:
    :param email:
    :param mindate: str
    :param maxdate: str
    :return:
    """
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=10000,
            email=email,
            mindate=mindate,
            maxdate=maxdate,
        )
        record = Entrez.read(handle)
    except RuntimeError:
        return -1, []
    except IncompleteRead:
        return -2, []
    except HTTPError:
        return -3, []


    return int(record["Count"]), record["IdList"]


def format_date(date: str) -> str:
    """Pubmed date format is YYYY/MM/DD. This function converts a date to this format.
    Sometimes the date can be in DD/MM/YYYY format, so this function converts it to YYYY/MM/DD format.
    It also replaces the "-" with "/".
    """
    components = date.replace("-", "/").split("/")

    if len(components[0]) == 2:
        formatted_date = components[2] + "/" + components[1] + "/" + components[0]
    else:
        formatted_date = date.replace("-", "/")

    return formatted_date


def safe_get_value(_dictionary: dict, column_name: str, default_value: str = "") -> str:
    """
    Return the value of a _dictionary key if the key exists, otherwise return the default value.
    """
    if column_name in _dictionary:
        return _dictionary[column_name]
    else:
        return default_value


def process_queries(
    df: pd.DataFrame,
    collection_stats: pd.DataFrame,
    queries,
    default_dates,
    email: str,
    verbose: bool = False,
) -> dict[str, dict[str, dict[str, int]]]:
    output_dicts = {query: {} for query in queries}

    for index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Processing topics for {len(queries)} queries",
    ):
        #collection_stats_row = collection_stats[collection_stats["id"] == row["id"]]
        #date_from = format_date(collection_stats_row["Date_from"].to_list()[0])
        #if not date_from:
        date_from = default_dates[0]
        #date_to = format_date(collection_stats_row["Date_to"].to_list()[0])
        #if not date_to:
        date_to = default_dates[1]

        if verbose:
            print(f"{row['id']=}")

        for query_name, query_column_mapping in queries.items():
            if pd.isna(row[query_column_mapping]):
                continue
            count, id_list = temporal_submission(
                query=row[query_column_mapping],
                email=email,
                mindate=date_from,
                maxdate=date_to,
            )
            if verbose:
                print(f"{query_name=}\t{count=}")

            output_dicts[query_name][str(row["id"])] = {
                str(pmid): 1 for pmid in id_list
            }

    return output_dicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries_file",
        type=str,
        default="../../data/1-queries/seed/Seed_zephyr_2426957.csv",
        help="Input file name",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../../data/2-runs/seed/",
        help="Output folder path",
    )

    parser.add_argument(
        "--stats_file",
        type=str,
        default="../../input/Seed/overall_collection.jsonl",
        help="Path to the collection stats file",
    )

    parser.add_argument(
        "--email", type=str, default="test@test.com", help="Email address for the query"
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs=2,
        default=("1946/01/01", "2018/12/31"),
        help="Date range for the query",
    )

    args = parser.parse_args()

    collection_stats = pd.read_json(args.stats_file, lines=True)

    run_name = args.queries_file.split("/")[-1][:-4]

    output_folder = f"{args.output_path}/{run_name}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(args.queries_file)

    queries = {
        "query": "query",
        "edited-search": "edited_search",
        "q1": "q1_answer",
        "q2": "q2_answer",
        "q3": "q3_answer",
        "q4": "q4_answer",
        "q5": "q5_answer",
        "related_q4": "related_q4_answer",
        "related_q5": "related_q5_answer",

        "guided_query": "guided_query_answer",
    }

    out_filename = f"{run_name}.json"

    output_dicts = process_queries(
        df,
        collection_stats,
        default_dates=args.dates,
        queries=queries,
        email=args.email,
        verbose=False,
    )

    runs = [
        Run(_returned_docs, name=f"{run_name}_{_query_type}")
        for _query_type, _returned_docs in output_dicts.items()
    ]

    for run in runs:
        run.save(f"{output_folder}/{run.name}.trec", kind="trec")
