import argparse
import os
from subprocess import Popen, PIPE

import pandas as pd
from tqdm import tqdm


def split_filename(filename: str) -> tuple[str, str, str, str]:
    """Splits a filename into:
    'collection'_'model'_seed_'query-type'.trec

    :param filename:
    :return:
    """
    parts = filename.split("_")
    collection = parts[0]
    model = parts[1]
    seed = parts[2]
    query_type = "_".join(parts[3:]).split(".")[0]
    return collection, model, seed, query_type

def run_trec_eval(
        trec_eval: str, qrels_file: str, run_file: str, metrics: list[str]
) -> dict:
    result_dict = {}
    command = (
            trec_eval
            + " "
            + " ".join([f"-m {metric}" for metric in metrics])
            + " "
            + qrels_file
            + " "
            + run_file
    )
    results = Popen(
        command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True
    ).stdout.readlines()
    for result in results:
        items = result.split()
        if (len(items) == 3) and (items[1] == "all"):
            result_dict[items[0]] = float(items[-1])
    return result_dict


def evaluate_with_trec_eval(trec_eval, qrels_file, run_file) -> dict:
    metrics = [["set_P", "set_recall", "set_F.1"], ["set_F.3"], ["set_F.05"]]
    results = {}
    for metric in metrics:
        results.update(run_trec_eval(trec_eval, qrels_file, run_file, metric))

    results["recall"] = results["set_recall"]
    results["precision"] = results["set_P"]
    results["f1"] = results["set_F_1"]
    results["f3"] = results["set_F_3"]
    results["f05"] = results["set_F_05"]
    results.pop("set_recall")
    results.pop("set_P")
    results.pop("set_F_1")
    results.pop("set_F_3")
    results.pop("set_F_05")
    return results


def evaluate(run_file: str, qrels_file: str, output_file: str):
    qrels_df = pd.read_csv(
        qrels_file,
        sep=" ",
        header=None,
        names=["query_id", "unused", "doc_id", "relevance"],
    )
    run_df = pd.read_csv(
        run_file,
        sep=" ",
        header=None,
        names=["query_id", "unused", "doc_id", "rank", "score", "run_name"],
    )
    run_name = run_df["run_name"].unique()[0]

    run_df["relevant"] = run_df.apply(
        lambda x: 1
        if any(
            (qrels_df["query_id"] == x["query_id"])
            & (qrels_df["doc_id"] == x["doc_id"])
            & (qrels_df["relevance"] > 0)
        )
        else 0,
        axis=1,
    )

    eval_results = []
    for query_id in tqdm(qrels_df["query_id"].unique()):
        y_true = set(qrels_df[qrels_df["query_id"] == query_id]["doc_id"].unique())
        y_pred = set(run_df[run_df["query_id"] == query_id]["doc_id"].unique())

        true_positives = y_pred.intersection(y_true)

        if len(true_positives) == 0:
            precision = 0
            recall = 0
            f1 = 0
            f3 = 0
            f05 = 0
        else:
            precision = len(true_positives) / len(y_pred)
            recall = len(true_positives) / len(y_true)
            if precision + recall == 0:
                f1 = 0
                f3 = 0
                f05 = 0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
                f3 = 10 * (precision * recall) / (9 * precision + recall)
                f05 = 1.25 * (precision * recall) / (0.25 * precision + recall)

        eval_results.append(
            {
                "query_id": query_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f3": f3,
                "f05": f05,
            }
        )

    results_df = pd.DataFrame(eval_results)

    trec_eval_res = evaluate_with_trec_eval(
        trec_eval="trec_eval", qrels_file=qrels_file, run_file=run_file
    )
    trec_eval_res["query_id"] = "trec_eval_all"

    results_df = pd.concat(
        [results_df, pd.DataFrame([trec_eval_res])], ignore_index=True
    )
    results_df["run_name"] = run_name

    results_df.to_csv(output_file, sep=",", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_folder",
        type=str,
        default="../../data/2-runs/seed/",
        help="Input folder path",
    )
    parser.add_argument(
        "--qrels_file",
        type=str,
        default="../../data/0-qrels/seed.qrels",
        help="Qrels file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../../data/3-evaluated/seed/",
        help="Output folder path",
    )
    args = parser.parse_args()

    for dirpath, _, filenames in os.walk(args.run_folder):
        relative_path = os.path.relpath(dirpath, args.run_folder)
        for filename in filenames:
            if filename.endswith(".trec"):
                output_file_dir = os.path.join(args.output_path, relative_path)
                if not os.path.exists(output_file_dir):
                    os.makedirs(output_file_dir)
                output_file = os.path.join(
                    output_file_dir, f"{filename[:-5]}.csv"
                )
                run_file_path = os.path.join(dirpath, filename)

                evaluate(
                    run_file=run_file_path,
                    qrels_file=args.qrels_file,
                    output_file=output_file,
                )
