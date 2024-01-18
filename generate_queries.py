from openai import OpenAI
import os
import json
import pandas as pd
import argparse


#dataset = "CSMeD-subset"

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
def generate_prompt(type="q1", title="", example_title="", example_query=""):
  system_input = ""
  user_input = ""
  if type == "q1":
    user_input = f"For a systematic review titled “{title}”, can you generate a systematic review Boolean query to find all included studies on PubMed for the review topic? Structure the output as a JSON with the field boolean_query."
  elif type == "q2":
    system_input = "You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. "
    user_input = f"Now you have your information need to conduct research on {title}. Please construct a highly effective systematic review Boolean query that can best serve your information need. Structure the output as a JSON with the field boolean_query."
  elif type == "q3":
    user_input = f"Imagine you are an expert systematic review information specialist; now you are given a systematic review research topic, with the topic title “{title}”. Your task is to generate a highly effective systematic review Boolean query to search on PubMed (refer to the professionally made ones); the query needs to be as inclusive as possible so that it can retrieve all the relevant studies that can be included in the research topic; on the other hand, the query needs to retrieve fewer irrelevant studies so that researchers can spend less time judging the retrieved documents. Structure the output as a JSON with the field boolean_query"
  elif type == "q4":
    system_input = f"You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. You are able to take an information need such as: \“{example_title}\” and generate valid pubmed queries such as: \“{example_query}\"."
    user_input = f"Now you have the information need to conduct research on “{title}”, please generate a highly effective systematic review Boolean query for the information need. Structure the output as a JSON with the field boolean_query."
  elif type == "q5":
    system_input = f"You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. A professional information specialist will extract PICO elements from information needs in a common practice in constructing a systematic review Boolean query. PICO means Patient/ Problem, Intervention, Comparison and Outcome. PICO is a format for developing a good clinical research question prior to starting one’s research. It is a mnemonic used to describe the four elements of a sound clinical foreground question. You are able to take an information need such as: \“{example_title}\" and you generate valid pubmed queries such as: \“{example_query}\"."
    user_input = f" Now you have your information need to conduct research on “{title}”. First, extract PICO elements from the information needs and construct a highly effective systematic review Boolean query that can best serve your information need. Structure the output as a JSON with the field boolean_query."


  return [system_input,user_input]

def create_querying_dataset(df, dataset='CSMeD'):
  example_title = "Thromboelastography (TEG) and rotational thromboelastometry (ROTEM) for trauma-induced coagulopathy in adult trauma patients with bleeding"
  example_query = "(Thrombelastography[mesh:noexp] OR (thromboelasto*[All Fields] OR thrombelasto*[All Fields] OR ROTEM[All Fields] OR “tem international”[All Fields] OR (thromb*[All Fields] AND elastom*[All Fields]) OR (rotational[All Fields] AND thrombelast[All Fields])) OR (Thrombelastogra*[All Fields] OR Thromboelastogra*[All Fields] OR TEG[All Fields] OR haemoscope[All Fields] OR haemonetics[All Fields] OR (thromb*[All Fields] AND elastogra*[All Fields])))"
  if dataset == 'CSMeD':
    query_dataframe = df[['title', 'abstract', 'review_type','review_id']]
  if dataset == 'Seed':
    print(df.columns)
    query_dataframe = df[["id", 'title', 'link_to_review', 'query', 'edited_search', 'seed_studies', 'included_studies']]
  query_dataframe['q1'] = query_dataframe.apply(lambda row: generate_prompt("q1", row['title']), axis=1)
  query_dataframe['q2'] = query_dataframe.apply(lambda row: generate_prompt("q2", row['title']), axis=1)
  query_dataframe['q3'] = query_dataframe.apply(lambda row: generate_prompt("q3", row['title']), axis=1)
  query_dataframe['q4'] = query_dataframe.apply(lambda row: generate_prompt("q4", row['title'], example_title, example_query), axis=1)
  query_dataframe['q5'] = query_dataframe.apply(lambda row: generate_prompt("q5", row['title'], example_title, example_query), axis=1)

  return query_dataframe

def generate_query(input, model = "gpt-3.5-turbo", seed=11777768):
  client = OpenAI()
  # models = ["gpt-3.5-turbo", "gpt-4"]
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    #model="gpt-4",
    seed=11777768,
    response_format= {"type": "json_object"},
    messages=[
      {"role": "system", "content": input[0]},
      {"role": "user", "content": input[1]}
    ]
  )
  try:
    answer = json.loads(completion.choices[0].message.content)
    print(answer)
    return answer['boolean_query']
  except json.JSONDecodeError as e:
    print(f'Error decoding JSON: {e}')
    return "Error decoding JSON"
  except KeyError as e:
    print(f'Error extracting value: {e}')
    return "Error extracting value"


def main(CSMeD=False, Seed=False):

  seeds = [2426957, 3007195, 4187709, 4366962, 5682402, 5915503, 7486832, 8486927, 8701227, 9143138]
  models = ["gpt-3.5-turbo-1106"] #"gpt-4-1106-preview"]:

  if CSMeD:
    df = read_CSMed()

    query_dataframe = create_querying_dataset(df, dataset="CSMeD")

    for seed in seeds:
      for model in models:
        query_dataframe['q1_answer'] = query_dataframe.apply(lambda row: generate_query(row['q1'],model=model, seed= seed), axis=1)
        print("q1_finished")
        query_dataframe['q2_answer'] = query_dataframe.apply(lambda row: generate_query(row['q2'],model=model, seed= seed), axis=1)
        print("q2_finished")
        query_dataframe['q3_answer'] = query_dataframe.apply(lambda row: generate_query(row['q3'],model=model, seed= seed), axis=1)
        print("q3_finished")
        query_dataframe['q4_answer'] = query_dataframe.apply(lambda row: generate_query(row['q4'],model=model, seed= seed), axis=1)
        print("q4_finished")
        query_dataframe['q5_answer'] = query_dataframe.apply(lambda row: generate_query(row['q5'],model=model, seed= seed), axis=1)
      query_dataframe.to_csv(f"output/CSMeD_f{model}_{seed}.csv")

  if Seed:
    df = read_Seed()

    query_dataframe = create_querying_dataset(df, dataset="Seed")

    for seed in seeds:
      for model in models:
        query_dataframe['q1_answer'] = query_dataframe.apply(lambda row: generate_query(row['q1'],model=model, seed= seed), axis=1)
        print("q1_finished")
        query_dataframe['q2_answer'] = query_dataframe.apply(lambda row: generate_query(row['q2'],model=model, seed= seed), axis=1)
        print("q2_finished")
        query_dataframe['q3_answer'] = query_dataframe.apply(lambda row: generate_query(row['q3'],model=model, seed= seed), axis=1)
        print("q3_finished")
        query_dataframe['q4_answer'] = query_dataframe.apply(lambda row: generate_query(row['q4'],model=model, seed= seed), axis=1)
        print("q4_finished")
        query_dataframe['q5_answer'] = query_dataframe.apply(lambda row: generate_query(row['q5'],model=model, seed= seed), axis=1)
      query_dataframe.to_csv(f"output/Seed_{model}_{seed}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset_path", type=str, required=True)
    #parser.add_argument("--email", type=str, default="tester@gmail.com")
    #parser.add_argument("--out_folder", type=str, default="output/")
    args = parser.parse_args()
    main(False, True)

