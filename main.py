import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import faiss
import pickle
from scipy.spatial import distance

directory = 'CSV_files'
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

llm = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

for csv_file in csv_files:
    csv_file = os.path.join(directory, csv_file)
    df = pd.read_csv(csv_file)

    if 'tweet_text' not in df.columns:
        if 'Tweet text' in df.columns:
            df.rename(columns={'Tweet text': 'tweet_text'}, inplace=True)
        else:
            print(f"No 'tweet_text' or 'Tweet text' column in {csv_file}. Skipping this file.")
            continue

    base_name = os.path.basename(csv_file).split('.')[0]
    pickle_file = f'{base_name}_embeddings.pkl'

    # Check if pickle file already exists
    if os.path.exists(pickle_file):
        print(f"Pickle file {pickle_file} loaded.")
        with open(pickle_file, 'rb') as f:
            index = pickle.load(f)
    else:
        print(f"Loading the Universal Sentence Encoder for {csv_file}...")
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        print("Encoder loaded successfully.")

        embeddings = []

        print("Generating embeddings for each tweet...")
        for tweet_text in df['tweet_text']:
            embedding = embed([tweet_text])
            embeddings.append(embedding.numpy()[0])
        print("Embeddings generated successfully.")

        if len(embeddings) == 0:
            print(f"No embeddings generated for {csv_file}. Skipping this file.")
            continue

        embeddings = np.array(embeddings)

        if embeddings.ndim == 1:
            print(f"Only one embedding generated for {csv_file}. Skipping this file.")
            continue

        dimension = embeddings.shape[1]  # Dimension of the embeddings
        print(f"Dimension of the embeddings: {dimension}")

        print("Creating FAISS index...")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print("FAISS index created successfully.")

        with open(pickle_file, 'wb') as f:
            pickle.dump(index, f)

        print(f"Embeddings stored in FAISS index and saved as pickle file: {pickle_file}")

    # User asks a question
    query = input("Please enter your question: ")
    query_embedding = llm([query]).numpy()[0]

    # Find the most similar tweets to the user's query
    D, I = index.search(np.array([query_embedding]), 5)  # Find the 5 nearest neighbors

    print("Here are the most similar tweets to your query:")
    for i in I[0]:
        print(df['tweet_text'].iloc[i])