import yaml
import os
from openai import OpenAI

OPENAI_API_KEY = "sk-proj-gwlzlAakpbEkKq0DuY3cT3BlbkFJyunclb2kN2K02lCsbxMN"
client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_full_guardrails(self):

    out_rails = {}

    with open("guardrails.yml", 'r') as file:
        guard_rails_raw = yaml.safe_load(file)

    for key, raw_rail in guard_rails_raw.items():
        embedded_texts = []
        out_rails[key] = {}
        for text in raw_rail['raw_texts']:
            embedded_texts.append(get_embedding(text))
        out_rails[key]['embedding_vectors'] = embedded_texts
        out_rails[key]['threshold'] = raw_rail['threshold']
    
    
    print(f"saving cached rails to: cache/guardrails.yaml")
    os.makedirs(os.path.dirname("cache/guardrails.yaml"), exist_ok=True)
    with open("cache/guardrails.yaml", 'w+') as f:
        yaml.dump(out_rails, f)


if __name__ == "__main__":
    get_full_guardrails()
