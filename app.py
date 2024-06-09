#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from openai import OpenAI
from pinecone import Pinecone
import config

# In[3]:


import os
from flask import Flask, request,redirect, url_for, send_from_directory, render_template, jsonify


# In[4]:


os.environ['PINECONE_API_KEY'] = config.pinecone_api
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

index = pc.Index('rag-spotify-vector')
index.describe_index_stats()


# In[5]:


#creating  OpenAi Client
os.environ['OPENAI_API_KEY']=config.openai_api
client =  OpenAI()


# In[6]:


#Embeding function
def get_embeddings(text, model='text-embedding-ada-002'):
  text = text.replace('\n',' ')
  return client.embeddings.create(input='text',model=model).data[0].embedding


# In[7]:


def get_context(query, embed_model='text-embedding-ada-002', k=5):
  query_embeddings = get_embeddings(query, model=embed_model)
  pinecone_res=index.query(vector=query_embeddings, top_k=k, include_metadata=True)
  contexts =[item['metadata']['text'] for item in pinecone_res['matches']]
  return contexts,query


# In[8]:


def augmented_query(user_query, embed_model='text-embedding-ada-002',k=5):
  contexts, query = get_context(user_query, embed_model=embed_model, k=k)

  return "\n\n---\n\n".join(contexts)+"\n\n---\n\n"+query


# In[9]:


primer = f"""
  you are a Question and Anwer bot.
  A highly intelligent system that answers user questions based on
  information provided by the user above each question.
  If the anser cannot be found in the infromation provided by the user, you truthfully
  answer " I do not know"
  """


# In[10]:


import textwrap
def ask_gpt(user_prompt, system_prompt = primer, model="gpt-3.5-turbo"):

  temperature_=0.7

  completion = client.chat.completions.create(
    model=model,
    temperature=temperature_,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
  )
  lines = (completion.choices[0].message.content).split("\n")
  lists = (textwrap.TextWrapper(width=90, break_long_words=False).wrap(line) for line in lines)
  return "\n".join("\n".join(list) for list in lists)


# ### API Functions

# In[11]:


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    if request.json and 'message' in request.json:
        user_input = request.json['message']
        result = ask_gpt(user_input)
        response = generate_response(result)
        return jsonify({'response': response})
    else:
        return jsonify({'response': "No message received"})


def generate_response(message):
    # Simple echo bot for demonstration; replace with actual logic
    return f"{message}"

if __name__=="__main__":
  app.run(debug=True, host='0.0.0.0')
