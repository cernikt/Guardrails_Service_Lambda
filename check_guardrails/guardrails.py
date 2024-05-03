import numpy as np
from numpy.linalg import norm
import json
import yaml
import os
from openai import OpenAI


class GuardRails:

    def __init__(self, cache_filepath):
        self.guardrails_cache_filepath = cache_filepath
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        # try and load cache path
        # doesnt work load with raw guard rails
        # save that to cache whereever that may be
        
        with open(self.guardrails_cache_filepath, 'r') as file:
            self.guard_rails_embeddings = yaml.safe_load(file)
        self.full_guardrails = self.guard_rails_embeddings


    def get_embedding(self, text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input = [text], model=model).data[0].embedding

    def check_guard_rails(self, input_text):
        embedded_user_q = self.get_embedding(input_text)
        return {key: self.matches_guard_rail(embedded_user_q, guard_rail, key) for key, guard_rail in self.full_guardrails.items()}
    
    def _default_match_func(self, cosine_sim, threshold):
        # checks if the max is above the threshold - if yes matches rail
        max_sim = max(cosine_sim)
        return max_sim > threshold

    def matches_guard_rail(self, embedded_user_query, guard_rail, key):
        cosine_distances = []
        
        for embedding in guard_rail['embedding_vectors']:
            cosine = np.dot(embedded_user_query, embedding)/(norm(embedded_user_query)*norm(embedding))
            cosine_distances.append(cosine)

        cosine_max = max(cosine_distances)
        print(f"key: {key}, max match: {cosine_max}, threshold: {guard_rail['threshold']}")
        return self._default_match_func(cosine_distances, guard_rail['threshold'])
    

def check_guardrails(event, context):
    # get the input text from the event
    input_text = event['text']
    # create the guardrails object
    guardrails = GuardRails("cache/guardrails.yaml")
    # check the guardrails
    guardrails_result = guardrails.check_guard_rails(input_text)
    # return the guardrails result
    return guardrails_result

# Need to figure out a way to store the guardrails in the cache in a vector database or something similar.
# This way we dont ahve to load the guardrails every time we run the lambda function
# Then the query time will actually be doable and not 500+ seconds

# How would I do that?
# Few options: PostgreSQL with the pgvector extension
# Aurora Serverless PostgreSQL with pgvector
# Amazon RDS
# All of these options would work, but I didnt have time to finish it. :(

