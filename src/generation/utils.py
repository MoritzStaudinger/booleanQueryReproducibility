import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import os

def extract_seed(url):
    # Fetch the HTML content of the website
    response = requests.get(url)
    html_content = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the <div> with id "eng-abstract"
    eng_abstract_div = soup.find('div', id='eng-abstract')

    if eng_abstract_div:
        # Find the <p> tag within the "eng-abstract" div
        paragraph_tag = eng_abstract_div.find('p')

        if paragraph_tag:
            # Extract the text content of the <p> tag
            paragraph_content = paragraph_tag.get_text(strip=True)
            return paragraph_content

    return None

def get_similarities():
    f = open('input/topic-similarity.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    df = pd.DataFrame(
        {'id': [], 'example_Seed_1': [], 'example_Seed_2': [], 'example_CLEF_1': [], 'example_CLEF_2': []})
    counter = 0

    for id_key in data.keys():
        first_numbers = []
        first_CD_values = []
        for key in data[id_key]:
            if key.isdigit():
                first_numbers.append(key)
            elif key.startswith('CD'):
                first_CD_values.append(key)
        if (len(first_CD_values) >= 2 and len(first_numbers) >= 2):
            df.loc[counter] = [id_key, first_numbers[0], first_numbers[1], first_CD_values[0], first_CD_values[1]]
            counter = counter + 1
    return df

def read_CSMed(directory="input/CSMeD-subset/"):
  data_frames = []
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith("review_details.json"):
        file_path = os.path.join(root, file)
        with open(file_path, 'r') as f:
          data = json.load(f)
          # df = pd.json_normalize(data, max_level=0)
          df = pd.json_normalize(data, max_level=1)
          data_frames.append(df)
  return pd.concat(data_frames, ignore_index=True)

def read_Seed(directory="input/Seed/"):
  return pd.read_json(directory+"overall_collection.jsonl", lines=True)

def read_CLEF(directory="input/CLEF_TAR"):
  file_path = directory+"/"+"all.title"
  ids = []
  titles = []
  with open(file_path, 'r') as file:
    for line in file:
      # Split the line into CD number and title
      id, title = line.strip().split(' ', 1)
      ids.append(id)
      titles.append(title)

  df = pd.DataFrame({'id': ids, 'title': titles})

  ##### Load 2018 files

  ids = []
  titles = []
  queries = []
  result_ids = []

  # Loop through files in the directory
  for filename in os.listdir(directory+"/2018/"):
      file_path = os.path.join(directory+"/2018/", filename)

      # Read the file line by line
      with open(file_path, 'r') as file:
          id = None
          title = None
          query_lines = []
          is_query_section = False

          for line in file:
              if line.startswith("Topic: CD"):
                  id = line.split()[1].strip()
              elif line.startswith("Title:"):
                  title = line[7:].strip()
              elif line.startswith("Query:"):
                  is_query_section = True
              elif line.startswith("Pids:"):
                  is_query_section = False
              elif is_query_section:
                  query_lines.append(line.strip())

          # Combine query lines into a single string
          query = '\n'.join(query_lines)

          # Append data to lists
          ids.append(id)
          titles.append(title)
          queries.append(query)

  # Create a Pandas DataFrame
  df_2018 = pd.DataFrame({'id': ids, 'title': titles, 'query': queries})

  df = df.merge(df_2018, how="outer") #not necessary to specify merge columns, as pandas use same named ones
  # Create a Pandas DataFrame
  return df


import os
import pandas as pd

def load_and_merge_files(folder_path, output_folder):
    # Create an empty dictionary to store dataframes for each random seed
    seed_dataframes = {}
    # Iterate through all files and directories in the specified folder
    for root, dirs, files in os.walk(folder_path):
        print(root)
        for csv_file in files:
            # Check if the file is a CSV file
            if csv_file.endswith('.csv'):
                # Extract random seed and optional information from the file name
                filename_array = csv_file.split('_')
                optional_info = ""
                if len(filename_array) > 3:
                    model_name = filename_array[1]
                    random_seed = filename_array[2]
                    optional_info = filename_array[3]
                    optional_info = optional_info.split('.')[0]  # Remove the '.csv' extension
                else :
                    model_name = filename_array[1]
                    random_seed = filename_array[2].split('.')[0]

                # Form the key for the dictionary using random seed
                key = f"{model_name}_{random_seed}"
                # Read the CSV file into a pandas dataframe
                df = pd.read_csv(os.path.join(root, csv_file))

                # Check the fourth variable in the file name and rename columns accordingly
                if optional_info == 'q4':
                    # Rename columns with "related" prefix
                    df.rename(columns={"q4_answer": "related_q4_answer", "q5_answer": "related_q5_answer"}, inplace=True)
                    columns = [col for col in df.columns if col not in ['id','related_q4_answer', 'related_q5_answer']]
                    df.drop(columns=columns, inplace=True)
                    print(df.head())
                elif optional_info == 'guided':
                    # Rename columns with "guided" prefix
                    columns = [col for col in df.columns if col not in ['id', 'guided_query_answer']]
                    df.drop(columns=columns, inplace=True)
                    df['guided_query_answer'] = df['guided_query_answer'].apply(transform_json)

                # Check if a dataframe for this random seed already exists in the dictionary
                if key in seed_dataframes:
                    # Merge the new dataframe with the existing one based on the "id" column
                    seed_dataframes[key] = pd.merge(seed_dataframes[key], df, on='id')
                else:
                    # If no dataframe exists, create a new one
                    seed_dataframes[key] = df

    # Save the resulting dataframes into separate CSV files
    for key, df in seed_dataframes.items():
        output_filename = f"Seed_{key}_meta.csv"
        output_path = os.path.join(output_folder, output_filename)
        df.to_csv(output_path, index=False)

def transform_json(json_str):
    try:
        json_str = json_str.replace("```json", "").replace("```", "")
        json_obj = json.loads(json_str)
        if isinstance(json_obj, dict) and "boolean_query" in json_obj:
            return json_obj["boolean_query"]
    except json.JSONDecodeError:
        pass  # Ignore JSON decoding errors, treat as regular string
    return json_str
