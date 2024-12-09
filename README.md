

## Installation

```zsh
conda create -n booleanRepro python=3.10
```

Install the requirements:

```zsh
$ pip install -r requirements.txt
```

Install trec_eval.



## Evaluation

First run search_query.py to generate the search results for each query.

```zsh
$ python src/evaluation/search_query.py
```

Then run evaluation.py to evaluate the search results.

```zsh
$ python src/evaluation/evaluation.py
```
