# EEGPT - Easy Elastic GPT

# This code is part of an Elastic Blog showing how to combine
# Elasticsearch's search relevancy power with
# OpenAI's GPT's Question Answering power
# https://www.elastic.co/blog/chatgpt-elasticsearch-openai-meets-private-data

# Code is presented for demo purposes but should not be used in production
# You may encounter exceptions which are not handled in the code

import openai
import sys
from elasticsearch import Elasticsearch

openai.api_key = '' #OpenAI API Key
es_pw = '' #Elastic PW
es_un = 'elastic'
es_id = '' #CloudID
model = "gpt-3.5-turbo-0301"

def es_connect():
    client = Elasticsearch(
        cloud_id=es_id,
        basic_auth=(es_un, es_pw)
        )
    return client

# Search ElasticSearch index and return body and URL of the result
def search(query_text):
    es = es_connect()
    if es is None:
        return None, None

    # Elasticsearch query (BM25) and kNN configuration for hybrid search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "title": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }],
            "filter": [{
                "exists": {
                    "field": "title-vector"
                }
            }]
        }
    }

    knn = {
        "field": "title-vector",
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text
            }
        },
        "boost": 24
    }

    fields = ["title", "body_content", "url"]
    index = 'search-elastic-docs'
    try:
        resp = es.search(index=index,
                         query=query,
                         knn=knn,
                         fields=fields,
                         size=1,
                         source=False)
    except Exception as e:
        print(f"[-] An error occurred while trying to search:\n[-] {e}")
        print("[-] Double check your Elastic connection strings")
        sys.exit(1)

    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]

    #print(f"[+] Sending the following query: {body}") #Prints the output of the query
    return body, url

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

# Generate a response from ChatGPT based on the given prompt
def chat_gpt(prompt, model="gpt-3.5-turbo", max_tokens=1024, max_context_tokens=4000, safety_margin=5):
    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(prompt, max_context_tokens - max_tokens - safety_margin)

    response = openai.ChatCompletion.create(model=model,
                                            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": truncated_prompt}])

    return response["choices"][0]["message"]["content"]


# Generate and display response
negResponse = "[-] I'm unable to answer the question based on the information I have from Elastic Docs."
# ^ Not sure if the above is needed anymore

query = input("[+] Enter your search... ")

resp, url = search(query)
prompt = f"I'm going to present you the following input: {query}. From this, I want you to summarize key points from the information only using this document: {resp}. If the input is a question, I want you to answer it. If it's a single word, summarize what it is. If I ask for steps, provide them."
answer = chat_gpt(prompt)

if negResponse in answer:
    print(f"[+] ChatGPT: {answer.strip()}")
else:
    print(f"[+] ChatGPT: {answer.strip()}\n\nDocs: {url}")