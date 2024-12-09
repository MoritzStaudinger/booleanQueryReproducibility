#!/bin/bash

# Specify the path to the folder containing the CSV files
input_folder="output/CLEF/Result/"

# Specify the path to the output folder
output_path="data/2-runs/clef/"

# Specify the path to the overall_collection.jsonl file
stats_file="input/Seed/overall_collection.jsonl"

# Loop through all CSV files in the input folder
for csv_file in "${input_folder}"CLEF_gpt-3.5-turbo-0125_4*.csv; do
    # Extract the filename without the path
    filename=$(basename -- "$csv_file")

    # Execute the Python command for each file
    python src/evaluation/search_query.py --queries_file "${input_folder}${filename}" --output_path "${output_path}" --stats_file "${stats_file}"
done