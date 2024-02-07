import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


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


def aggregate_evaluation(results_folder):
    results = []
    for dirpath, _, filenames in os.walk(results_folder):
        for filename in filenames:
            if filename.endswith(".csv"):
                collection, model, seed, query_type = split_filename(filename)
                df = pd.read_csv(os.path.join(dirpath, filename))
                df["collection"] = collection
                df["model"] = model
                df["seed"] = seed
                df["query_type"] = query_type
                results.append(df)
    return pd.concat(results, ignore_index=True)


def write_results_table(df):
    pass


if __name__ == "__main__":
    results_path = "data/3-evaluated/"

    st.set_page_config(layout="wide")
    st.title("Evaluation of boolean queries reproduction")

    df = aggregate_evaluation(results_path)

    # remove trec_eval_all
    df = df[df["query_id"] != "trec_eval_all"]

    st.dataframe(df)

    # sort queries alphabetically
    df["query_type"] = pd.Categorical(
        df["query_type"], categories=sorted(df["query_type"].unique())
    )

    metrics = ["precision", "recall", "f1", "f3"]
    n_metrics = len(metrics)

    st.sidebar.header("Filter by collection")
    collections = df["collection"].unique()
    collection = st.sidebar.selectbox("Select collection", collections)
    df = df[df["collection"] == collection]

    st.sidebar.header("Filter by model")
    models = df["model"].unique()
    model = st.sidebar.selectbox("Select model", models)
    df = df[df["model"] == model]

    st.header("Mean results grouped by query type")
    st.dataframe(df.groupby("query_type")[metrics].mean())

    if st.button("Export to LaTeX"):
        latex_string = (
            df.groupby("query_type")[metrics].mean().to_latex(index=True, header=True)
        )
        st.text_area("Copy the LaTeX string below:", latex_string, height=300)

    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 4))

    for idx, metric in enumerate(metrics):
        sns.boxplot(x="query_type", y=metric, data=df, ax=axes[idx])
        axes[idx].set_title(f"Box Plot of {metric.capitalize()}")
        axes[idx].set_xlabel("Query Type")
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90)

    plt.tight_layout()
    st.pyplot(fig)
