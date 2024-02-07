from langchain.memory import ConversationBufferMemory
from mistralai.exceptions import MistralException
from openai import OpenAI
import os
import json
import argparse
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI;
import utils


#dataset = "CSMeD-subset"


def generate_prompt_gpt(type="q1", title="", abstract="", example_title="", example_abstract="", example_query="", initial_query=""):
  system_input = ""
  user_input = ""
  if type == "q1":
    user_input = f"For a systematic review titled “{title}”, can you generate a systematic review Boolean query to find all included studies on PubMed for the review topic? Structure the output as a JSON with the field boolean_query and create the boolean query without filtering based on the year."
  elif type == "q2":
    system_input = "You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. "
    user_input = f"Now you have your information need to conduct research on {title}. Please construct a highly effective systematic review Boolean query that can best serve your information need. Structure the output as a JSON with the field boolean_query and create the boolean query without filtering based on the year."
  elif type == "q3":
    user_input = f"Imagine you are an expert systematic review information specialist; now you are given a systematic review research topic, with the topic title “{title}”. Your task is to generate a highly effective systematic review Boolean query to search on PubMed (refer to the professionally made ones); the query needs to be as inclusive as possible so that it can retrieve all the relevant studies that can be included in the research topic; on the other hand, the query needs to retrieve fewer irrelevant studies so that researchers can spend less time judging the retrieved documents. Structure the output as a JSON with the field boolean_query  and create the boolean query without filtering based on the year."
  elif type == "q4":
    system_input = f"You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. You are able to take an information need such as: \“{example_title}\” and generate valid pubmed queries such as: \“{example_query}\"."
    user_input = f"Now you have the information need to conduct research on “{title}”, please generate a highly effective systematic review Boolean query for the information need. Structure the output as a JSON with the field boolean_query and create the boolean query without filtering based on the year."
  elif type == "q5":
    system_input = f"You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. A professional information specialist will extract PICO elements from information needs in a common practice in constructing a systematic review Boolean query. PICO means Patient/ Problem, Intervention, Comparison and Outcome. PICO is a format for developing a good clinical research question prior to starting one’s research. It is a mnemonic used to describe the four elements of a sound clinical foreground question. You are able to take an information need such as: \“{example_title}\" and you generate valid pubmed queries such as: \“{example_query}\"."
    user_input = f" Now you have your information need to conduct research on “{title}”. First, extract PICO elements from the information needs and construct a highly effective systematic review Boolean query that can best serve your information need. Structure the output as a JSON with the field boolean_query and create the boolean query without filtering based on the year."
  elif type == "q6":
    user_input = f'For a systematic review seed Boolean query: "{initial_query}", This query retrieves too many irrelevant documents and too few relevant documents about the information need: “{title}”, Please correct this query so that it can retrieve fewer irrelevant documents and more relevant documents<s>'
  elif type == "q3_abstract":
    user_input = f"Imagine you are an expert systematic review information specialist; now you are given a systematic review research topic, with the topic title “{title}” and the abstract “{abstract}”. Your task is to generate a highly effective systematic review Boolean query to search on PubMed (refer to the professionally made ones); the query needs to be as inclusive as possible so that it can retrieve all the relevant studies that can be included in the research topic; on the other hand, the query needs to retrieve fewer irrelevant studies so that researchers can spend less time judging the retrieved documents. Structure the output as a JSON with the field boolean_query and create the boolean query without filtering based on the year."
  elif type == "q4_abstract":
    system_input = f" You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. You are able to take an information need such as a systematic review with the title: \“{example_title}\” and abstract: \"{example_abstract}\" and generate valid pubmed queries such as: \“{example_query}\""
    user_input = f" Now you have the information need to conduct research on a review with the title: “{title}” and the following abstract: “{abstract}”, please generate a highly effective systematic review Boolean query for the information need. Structure the output as a JSON with the field boolean_query and create the boolean query without filtering based on the year."
  elif type == "q5_abstract":
    system_input = f"You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. A professional information specialist will extract PICO elements from information needs in a common practice in constructing a systematic review Boolean query. PICO means Patient/ Problem, Intervention, Comparison and Outcome. PICO is a format for developing a good clinical research question prior to starting one’s research. It is a mnemonic used to describe the four elements of a sound clinical foreground question. You are able to take an information need such as a systematic review with the title: \“{example_title}\” and abstract: \"{example_abstract}\"and you generate valid pubmed queries such as: \“{example_query}\""
    user_input = f"Now you have your information need to conduct research on a review with the title: “{title}” and the following abstract: “{abstract}”. First, extract PICO elements from the information needs and construct a highly effective systematic review Boolean query that can best serve your information need. Structure the output as a JSON with the field boolean_query and create the boolean query without filtering based on the year"


  return [system_input,user_input]

def generate_prompt_mistral(type="q1", title="", abstract="", example_title="", example_query="", example_abstract="", initial_query=""):
  system_input = ""
  user_input = ""
  if type == "q1":
    user_input = f"<s>[INST] For a systematic review titled “{title}”, can you generate a systematic review Boolean query to find all included studies on PubMed for the review topic? Just generate the Boolean Query without explanations and without filtering based on the year[/INST]</s>"
  elif type == "q2":
    system_input = "<s>[INST] You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. "
    user_input = system_input + f"Now you have your information need to conduct research on {title}. Please construct a highly effective systematic review Boolean query that can best serve your information need. Just generate the Boolean Query without explanations and without filtering based on the year[/INST]</s>"

  elif type == "q3":
    user_input = f"<s>[INST] Imagine you are an expert systematic review information specialist; now you are given a systematic review research topic, with the topic title “{title}”. Your task is to generate a highly effective systematic review Boolean query to search on PubMed (refer to the professionally made ones); the query needs to be as inclusive as possible so that it can retrieve all the relevant studies that can be included in the research topic; on the other hand, the query needs to retrieve fewer irrelevant studies so that researchers can spend less time judging the retrieved documents. Just generate the Boolean Query without explanations and without filtering based on the year[/INST]</s>"
  elif type == "q4":
    system_input = f"<s>[INST] You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. You are able to take an information need such as: \“{example_title}\” and generate valid pubmed queries such as: [/INST]\“{example_query}\"</s>"
    user_input = system_input + f"[INST] Now you have the information need to conduct research on “{title}”, please generate a highly effective systematic review Boolean query for the information need. Just generate the Boolean Query without explanations and without filtering based on the year[/INST]"
  elif type == "q5":
    system_input = f"<s> [INST] You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. A professional information specialist will extract PICO elements from information needs in a common practice in constructing a systematic review Boolean query. PICO means Patient/ Problem, Intervention, Comparison and Outcome. PICO is a format for developing a good clinical research question prior to starting one’s research. It is a mnemonic used to describe the four elements of a sound clinical foreground question. You are able to take an information need such as: \“{example_title}\" and you generate valid pubmed queries such as: [/INST]\“{example_query}\"</s>"
    user_input = system_input + f"[INST] Now you have your information need to conduct research on “{title}”. First, extract PICO elements from the information needs and construct a highly effective systematic review Boolean query that can best serve your information need. Just generate the Boolean Query without explanations and without filtering based on the year[/INST]"
  elif type == "q6":
    user_input = f'<s>[INST]For a systematic review seed Boolean query: "{initial_query}", This query retrieves too many irrelevant documents and too few relevant documents about the information need: “{title}”, Please correct this query so that it can retrieve fewer irrelevant documents and more relevant documents [/INST]<s>'
  elif type == "q3_abstract":
    user_input = f"<s>[INST] Imagine you are an expert systematic review information specialist; now you are given a systematic review research topic, with the topic title “{title}” and the abstract “{abstract}”. Your task is to generate a highly effective systematic review Boolean query to search on PubMed (refer to the professionally made ones); the query needs to be as inclusive as possible so that it can retrieve all the relevant studies that can be included in the research topic; on the other hand, the query needs to retrieve fewer irrelevant studies so that researchers can spend less time judging the retrieved documents. Just generate the Boolean Query without explanations and without filtering based on the year[/INST]</s>"
  elif type == "q4_abstract":
    system_input = f"<s>[INST] You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. You are able to take an information need such as a systematic review with the title: \“{example_title}\” and abstract: \"{example_abstract}\" and generate valid pubmed queries such as: [/INST]\“{example_query}\"</s>"
    user_input = system_input + f"[INST] Now you have the information need to conduct research on a review with the title: “{title}” and the following abstract: “{abstract}”, please generate a highly effective systematic review Boolean query for the information need. Just generate the Boolean Query without explanations and without filtering based on the year[/INST]"
  elif type == "q5_abstract":
    system_input = f"<s> [INST] You are an information specialist who develops Boolean queries for systematic reviews. You have extensive experience developing highly effective queries for searching the medical literature. Your specialty is developing queries that retrieve as few irrelevant documents as possible and retrieve all relevant documents for your information need. A professional information specialist will extract PICO elements from information needs in a common practice in constructing a systematic review Boolean query. PICO means Patient/ Problem, Intervention, Comparison and Outcome. PICO is a format for developing a good clinical research question prior to starting one’s research. It is a mnemonic used to describe the four elements of a sound clinical foreground question. You are able to take an information need such as a systematic review with the title: \“{example_title}\” and abstract: \"{example_abstract}\"and you generate valid pubmed queries such as: [/INST]\“{example_query}\"</s>"
    user_input = system_input + f"[INST] Now you have your information need to conduct research on a review with the title: “{title}” and the following abstract: “{abstract}”. First, extract PICO elements from the information needs and construct a highly effective systematic review Boolean query that can best serve your information need. Just generate the Boolean Query without explanations and without filtering based on the year[/INST]"

  #elif type == "q7":
    #user_input = f'For a systematic review seed Boolean query: “{example_initial_query}", This query retrieves too many irrelevant documents and too few relevant documents about the information need: “{example_title}”, therefore it should be corrected to: “{example_refined_query}”. Now your task is to correct a systematic review Boolean query: "{initial_query}" for information need “{title}”, so it can retrieve fewer irrelevant documents and more relevant documents.'

  return user_input


def get_summary_from_seed(id):
  return utils.extract_seed("https://pubmed.ncbi.nlm.nih.gov/" + id + "/")


def get_example_title(query_dataframe, similar_docs, row, field = 'example_Seed_1'):
  relevant_seed_id = similar_docs[similar_docs['id'] == str(row['id'])][field]
  title = query_dataframe[query_dataframe['id'] == int(relevant_seed_id)]['title'].to_list()[0]
  return title


def get_example_query(query_dataframe, similar_docs, row, field='example_Seed_1'):
  relevant_seed_id = similar_docs[similar_docs['id'] == str(row['id'])][field]
  query = query_dataframe[query_dataframe['id'] == int(relevant_seed_id)]['query'].to_list()[0]
  return query



def create_querying_dataset(df, dataset='CSMeD', model="gpt", questions=['q1','q2'], related_example=False):
  if related_example:
    similar_docs = utils.get_similarities()
  else:
    example_title = "Thromboelastography (TEG) and rotational thromboelastometry (ROTEM) for trauma-induced coagulopathy in adult trauma patients with bleeding"
    example_query = "(Thrombelastography[mesh:noexp] OR (thromboelasto*[All Fields] OR thrombelasto*[All Fields] OR ROTEM[All Fields] OR “tem international”[All Fields] OR (thromb*[All Fields] AND elastom*[All Fields]) OR (rotational[All Fields] AND thrombelast[All Fields])) OR (Thrombelastogra*[All Fields] OR Thromboelastogra*[All Fields] OR TEG[All Fields] OR haemoscope[All Fields] OR haemonetics[All Fields] OR (thromb*[All Fields] AND elastogra*[All Fields])))"


  if dataset == 'CSMeD':
    query_dataframe = df[['title', 'abstract', 'doi', "decision",'review_id', 'document_id', 'PubMed ID']]
  if dataset == 'Seed':
    query_dataframe = df[["id", 'title', 'link_to_review', 'query', 'edited_search', 'seed_studies', 'included_studies']]
  if dataset == 'CLEF':
    query_dataframe = df[["id", 'title', 'query']]


  if "gpt" in model:
    for question in questions:
      if question == "guided_query":
        continue
      else:
        if related_example & (dataset == 'Seed'):
          query_dataframe[question] = query_dataframe.apply(
            lambda row: generate_prompt_gpt(type=question, title=row['title'], example_title=get_example_title(query_dataframe, similar_docs, row), example_query=get_example_query(query_dataframe, similar_docs, row)), axis=1)
        else:
          query_dataframe[question] = query_dataframe.apply(lambda row: generate_prompt_gpt(type=question, title=row['title'], example_title=example_title, example_query=example_query), axis=1)
  if "mistral" in model:
    for question in questions:
      if question == 'guided_query':
        continue
      else:
        if related_example & (dataset == 'Seed'):
          query_dataframe[question] = query_dataframe.apply(lambda row: generate_prompt_mistral(type=question, title=row['title'], example_title=get_example_title(query_dataframe, similar_docs, row), example_query=get_example_query(query_dataframe, similar_docs, row)), axis=1)
        else:
          query_dataframe[question] = query_dataframe.apply(lambda row: generate_prompt_mistral(type=question, title=row['title'], example_title=example_title, example_query=example_query), axis=1)

  return query_dataframe

def generate_query_gpt(input, model = "gpt-3.5-turbo", seed=11777768):
  client = OpenAI()
  # models = ["gpt-3.5-turbo", "gpt-4"]
  completion = client.chat.completions.create(
    model=model,
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

def generate_openai_memory_query(type="q1", title="", summary="", seed=11777768, model="gpt-3.5-turbo-1106"):
  input = [f"Follow my instructions precisely to develop a highly effective Boolean query for a medical systematic review literature search. Do not explain or elaborate. Only respond with exactly what I request. First, Given the following statement and text from a relevant study, please identify 50 terms or phrases that are relevant. The terms you identify should be used to retrieve more relevant studies, so be careful that the terms you choose are not too broad. You are not allowed to have duplicates in your list. statement: '{title}' Text: '{summary}'",
           f"For each item in the list you created in the step before, classify it into three categories: terms relating to health conditions (A), terms relating to a treatment (B), terms relating to types of study design (C). When an item does not fit one of these categories, mark it as (N/A). Each item needs to be categorised into (A), (B), (C), or (N/A).",
           "Using the categorised list you created in the step before, create a Boolean query that can be submitted to PubMed which groups together items from each category. For example: ((itemA1[Title/Abstract] OR itemA2[Title/Abstract] or itemA2[Title/Abstract]) AND (itemB1[Title/Abstract] OR itemB2[Title/Abstract] OR itemB3[Title/Abstract]) AND (itemC1[Title/Abstract] OR itemC2[Title/Abstract] OR itemC3[Title/Abstract]))",
           "Use your expert knowledge and the information from the steps prior to refine the query, making it retrieve as many relevant documents as possible while minimising the total number of documents retrieved. Also add relevant MeSH terms into the query where necessary, e.g., MeSHTerm[MeSH]. Retain the general structure of the query, however, with each main clause of the query corresponding to a PICO element. The final query still needs to be executable on PubMed, so it should be a valid query. Structure the output as a JSON with the field boolean_query and create the boolean query without filtering based on the year."]

  llm = ChatOpenAI(model_name=model, model_kwargs={"seed":seed})

  conversation = ConversationChain(
    llm=llm,
    verbose=False,
    memory=ConversationBufferMemory()
  )

  answer = ""
  for i in input:
    answer = conversation.predict(input=i)

  print(conversation.memory)
  return answer


def generate_mistral_memory_query(type="q1", title="", summary="", seed=11777768, model="mistral-tiny"):
  input = [
    f"<s>[INST]Follow my instructions precisely to develop a highly effective Boolean query for a medical systematic review literature search. Do not explain or elaborate. Only respond with exactly what I request. First, Given the following statement and text from a relevant study, please identify 50 terms or phrases that are relevant. The terms you identify should be used to retrieve more relevant studies, so be careful that the terms you choose are not too broad. You are not allowed to have duplicates in your list. statement: '{title}' Text: '{summary}'[/INST]",
    f"<s>[INST]For each item in the list you created in the step before, classify it into three categories: terms relating to health conditions (A), terms relating to a treatment (B), terms relating to types of study design (C). When an item does not fit one of these categories, mark it as (N/A). Each item needs to be categorised into (A), (B), (C), or (N/A).[/INST]",
    "<s>[INST]Using the categorised list you created in the step before, create a Boolean query that can be submitted to PubMed which groups together items from each category. For example: ((itemA1[Title/Abstract] OR itemA2[Title/Abstract] or itemA2[Title/Abstract]) AND (itemB1[Title/Abstract] OR itemB2[Title/Abstract] OR itemB3[Title/Abstract]) AND (itemC1[Title/Abstract] OR itemC2[Title/Abstract] OR itemC3[Title/Abstract]))[/INST]",
    "<s>[INST]Use your expert knowledge and the information from the steps prior to refine the query, making it retrieve as many relevant documents as possible while minimising the total number of documents retrieved. Also add relevant MeSH terms into the query where necessary, e.g., MeSHTerm[MeSH]. Retain the general structure of the query, however, with each main clause of the query corresponding to a PICO element. The final query still needs to be executable on PubMed, so it should be a valid query. Return only the boolean query, without any additional information and without filtering based on the year.[/INST]"]

  api_key = os.environ["MISTRAL_API_KEY"]

  client = MistralClient(api_key=api_key)

  messages = []

  try:
    completion = ""
    for i in input:
     messages.append(ChatMessage(role="user", content=i))
     completion = client.chat(model=model,messages=messages,random_seed=seed)
     messages.append(completion.choices[0].message)
     print(completion.choices[0].message.content.replace("\n", " "))


    answer = completion.choices[0].message.content.replace("\n", " ")
  except MistralException as e:
    answer = f"Error: {e}"

  #answer = completion.choices[0].message.content.replace("\n", " ")
  return answer



def generate_gpt_memory_query(type="q1", title="", summary="", seed=11777768, model="gpt-3.5-turbo-1106"):
  input = [
    f"Follow my instructions precisely to develop a highly effective Boolean query for a medical systematic review literature search. Do not explain or elaborate. Only respond with exactly what I request. First, Given the following statement and text from a relevant study, please identify 50 terms or phrases that are relevant. The terms you identify should be used to retrieve more relevant studies, so be careful that the terms you choose are not too broad. You are not allowed to have duplicates in your list. statement: '{title}' Text: '{summary}'",
    f"For each item in the list you created in the step before, classify it into three categories: terms relating to health conditions (A), terms relating to a treatment (B), terms relating to types of study design (C). When an item does not fit one of these categories, mark it as (N/A). Each item needs to be categorised into (A), (B), (C), or (N/A).",
    "Using the categorised list you created in the step before, create a Boolean query that can be submitted to PubMed which groups together items from each category. For example: ((itemA1[Title/Abstract] OR itemA2[Title/Abstract] or itemA2[Title/Abstract]) AND (itemB1[Title/Abstract] OR itemB2[Title/Abstract] OR itemB3[Title/Abstract]) AND (itemC1[Title/Abstract] OR itemC2[Title/Abstract] OR itemC3[Title/Abstract]))",
    "Use your expert knowledge and the information from the steps prior to refine the query, making it retrieve as many relevant documents as possible while minimising the total number of documents retrieved. Also add relevant MeSH terms into the query where necessary, e.g., MeSHTerm[MeSH]. Retain the general structure of the query, however, with each main clause of the query corresponding to a PICO element. The final query still needs to be executable on PubMed, so it should be a valid query. Structure the output as a JSON with the field boolean_query and create the boolean query without filtering based on the year."]

  client = OpenAI()
  # models = ["gpt-3.5-turbo", "gpt-4"]
  completion = client.chat.completions.create(
    model=model,
    # model="gpt-4",
    seed=11777768,
    response_format={"type": "json_object"},
    messages=[
      {"role": "system", "content": input[0]},
      {"role": "user", "content": input[1]}
    ]
  )

  for i in input:
    completion.append()
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

def generate_query_mistral(input, model = "mistral_tiny", seed=11777768):
  api_key = os.environ["MISTRAL_API_KEY"]

  client = MistralClient(api_key=api_key)

  messages = [
    ChatMessage(role="user", content=input)
  ]

  try:
    completion = client.chat(
      model=model,
      messages=messages,
      random_seed=seed
    )
    answer = completion.choices[0].message.content.replace("\n", " ")
  except MistralException as e:
    answer = f"Error: {e}"

  #answer = completion.choices[0].message.content.replace("\n", " ")
  print(answer)
  return answer

def main(CSMeD=False, Seed=False, CLEF=False, models = ["gpt-3.5-turbo-1106"]):
#  2426957, 3007195, 9143138, 4187709, 4366962, 5682402, 5915503, 7486832, 8486927, 8701227  #created an error for mistral
  seeds = [3007195, 9143138, 4187709, 4366962, 5682402, 5915503, 7486832, 8486927, 8701227]
  #"gpt-4-1106-preview"]:
  questions = ['q1','q2','q3','q4', 'q5', 'q3_abstract','q4_abstract','q5_abstract']
  #questions = ['q4', 'q5']
  #questions = ['q3_abstract','q4_abstract','q5_abstract']
  #questions = ['guided_query']

  if CSMeD:
    df = utils.read_CSMed()


    for seed in seeds:
      for model in models:
        query_dataframe = create_querying_dataset(df, dataset="CSMeD", model=model, questions=questions, related_example=False)
        for question in questions:
          if "mistral" in model:
            query_dataframe[f'{question}_answer'] = query_dataframe.apply(
              lambda row: generate_query_mistral(row[question], model=model, seed=seed), axis=1)
          else:
            query_dataframe[f'{question}_answer'] = query_dataframe.apply(lambda row: generate_query_gpt(row[question],model=model, seed= seed), axis=1)
          print(f"{question} finished")
      query_dataframe.to_csv(f"output/CSMeD_f{model}_{seed}_{questions[0]}.csv")

  if Seed:
    df = utils.read_Seed()
    for seed in seeds:
      for model in models:
        query_dataframe = create_querying_dataset(df, dataset="Seed", model=model, questions=questions, related_example=True)
        for question in questions:
          if "mistral" in model:
            if question == 'guided_query':
              query_dataframe[f'{question}_answer'] = query_dataframe.apply(
                lambda row: generate_mistral_memory_query(question, row['title'],
                                                         get_summary_from_seed(row['seed_studies'][0]), seed=seed,
                                                         model=model), axis=1)
            else:
              query_dataframe[f'{question}_answer'] = query_dataframe.apply(
                lambda row: generate_query_mistral(row[question], model=model, seed=seed), axis=1)
          elif "gpt" in model:
            if question == 'guided_query':
              query_dataframe[f'{question}_answer'] = query_dataframe.apply(
                lambda row: generate_openai_memory_query(question, row['title'],
                                                         get_summary_from_seed(row['seed_studies'][0]), seed=seed, model=model), axis=1)
            else:
              query_dataframe[f'{question}_answer'] = query_dataframe.apply(lambda row: generate_query_gpt(row[question],model=model, seed= seed), axis=1)
          print(f"{question} finished")
      query_dataframe.to_csv(f"output/Seed_{model}_{seed}_{questions[0]}.csv")


  if CLEF:
    df = utils.read_CLEF()
    for seed in seeds:
      for model in models:
        query_dataframe = create_querying_dataset(df, dataset="CLEF", model=model, questions=questions, related_example=False)
        for question in questions:
          if "mistral" in model:
            if question == 'guided_query':
              raise NotImplementedError("guided query not here yet")
              #query_dataframe[f'{question}_answer'] = query_dataframe.apply(
               # lambda row: generate_mistral_memory_query(question, row['title'],
                #                                         get_summary_from_seed(row['seed_studies'][0]), seed=seed,
                #                                         model=model), axis=1)
            else:
              query_dataframe[f'{question}_answer'] = query_dataframe.apply(
                lambda row: generate_query_mistral(row[question], model=model, seed=seed), axis=1)
          else:
            if question == 'guided_query':
              raise NotImplementedError("guided query not here yet")
              #query_dataframe[f'{question}_answer'] = query_dataframe.apply(
                #lambda row: generate_openai_memory_query(question, row['title'],
                #                                         get_summary_from_seed(row['seed_studies'][0]), seed=seed, model=model), axis=1)
            else:
              query_dataframe[f'{question}_answer'] = query_dataframe.apply(lambda row: generate_query_gpt(row[question],model=model, seed= seed), axis=1)
          print(f"{question} finished")
      query_dataframe.to_csv(f"../../output/CLEF_{model}_{seed}_{questions[0]}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset_path", type=str, required=True)
    #parser.add_argument("--email", type=str, default="tester@gmail.com")
    #parser.add_argument("--out_folder", type=str, default="output/")
    args = parser.parse_args()
    main(True, False,False, models=['gpt-3.5-turbo-1106'])
    #main(False,False, True, models=['gpt-4-1106-preview'])
    #main(True, False, CLEF=False, models=['mistral-tiny'])

