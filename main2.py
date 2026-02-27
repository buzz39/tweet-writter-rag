import os
import pickle
import dill
import pandas as pd
import redis
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import openai
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback

directory = 'CSV_files'
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
vector_stores = {}
r = redis.StrictRedis(host=os.environ['REDIS_HOST'], port=6380, password=os.environ['REDIS_PASSWORD'], ssl=True)

for csv_file in csv_files:
    csv_file = os.path.join(directory, csv_file)
    df = pd.read_csv(csv_file)

    if 'tweet_text' not in df.columns:
        if 'Tweet text' in df.columns:
            df.rename(columns={'Tweet text': 'tweet_text'}, inplace=True)
        else:
            print(f"No 'tweet_text' or 'Tweet text' column in {csv_file}. Skipping this file.")
            continue

    # Check if 'tweet_text' column is empty
    if df['tweet_text'].isnull().all():
        print(f"'tweet_text' column in {csv_file} is empty. Skipping this file.")
        continue

    # Create a FAISS index file name based on the CSV file name
    index_file = csv_file.replace('.csv', '.index')

    # Check if embeddings are already stored in Redis
    if r.exists(csv_file):
        print(f"Loading embeddings from Redis for {csv_file}...")
        embeddings = pickle.loads(r.get(csv_file))
    else:
        print(f"Generating embeddings for {csv_file}...")
        embeddings = FAISS.from_texts(df['tweet_text'], embedding=embeddings)
        print("Embeddings generated successfully.")
        # Save the embeddings to Redis
        r.set(csv_file, dill.dumps(embeddings))

vector_stores[csv_file] = FAISS(embeddings)
# User asks a question
query = input("Please enter your question: ")

for csv_file, VectorStores in vector_stores.items():
    # Find the most similar tweets to the user's query
    docs = VectorStores.similarity_search(query=query, k=5)  # Find the 5 nearest neighbors

    print(f"Here are the most similar tweets to your query in {csv_file}:")
    for doc in docs:
        print(doc)

llm = openai.OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])
chain = load_qa_chain(llm=llm, chain_type="stuff")
with get_openai_callback() as cb:
    response = chain.run(input_documents=docs, question=query)
    print(cb)
print(response)