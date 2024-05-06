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

# Load the Universal Sentence Encoder
try:
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
except Exception as e:
    print(f"Failed to load the Universal Sentence Encoder: {e}")
    exit()

# Get list of CSV files
directory = 'CSV_files'
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Initialize lists for storing embeddings and tweets
all_embeddings = []
all_tweets = []

# Process each CSV file
for csv_file in csv_files:
    csv_file = os.path.join(directory, csv_file)
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Failed to load {csv_file}: {e}")
        continue

    # Check for the 'tweet_text' column
    if 'tweet_text' not in df.columns:
        if 'Tweet text' in df.columns:
            df.rename(columns={'Tweet text': 'tweet_text'}, inplace=True)
        else:
            print(f"No 'tweet_text' or 'Tweet text' column in {csv_file}. Skipping this file.")
            continue

    # Load or generate embeddings
    base_name = os.path.basename(csv_file).split('.')[0]
    pickle_file = f'{base_name}_embeddings.pkl'
    if os.path.exists(pickle_file):
        print(f"Loading embeddings from {pickle_file}...")
        try:
            with open(pickle_file, 'rb') as f:
                embeddings = pickle.load(f)
        except Exception as e:
            print(f"Failed to load embeddings from {pickle_file}: {e}")
            continue
    else:
        print(f"Generating embeddings for each tweet in {csv_file}...")
        embeddings = [embed([tweet_text]).numpy()[0] for tweet_text in df['tweet_text']]
        print("Embeddings generated successfully.")
        print(f"Saving embeddings to {pickle_file}...")
        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(embeddings, f)
        except Exception as e:
            print(f"Failed to save embeddings to {pickle_file}: {e}")

    # Add embeddings and tweets to the main lists
    all_embeddings.extend(embeddings)
    all_tweets.extend(df['tweet_text'].tolist())

# Convert the list of embeddings to a numpy array
all_embeddings = np.array(all_embeddings)

# Check if only one embedding was generated
if all_embeddings.ndim == 1:
    print(f"Only one embedding generated. Exiting.")
    exit()

# Create a FAISS index
dimension = all_embeddings.shape[1]  # Dimension of the embeddings
print(f"Dimension of the embeddings: {dimension}")
print("Creating FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings)
print("FAISS index created successfully.")

# Get a query from the user
query = input("Please enter your question: ")
query_embedding = embed([query]).numpy()[0]

# Find the most similar tweets to the user's query
num_neighbors = 5  # Number of nearest neighbors to find
D, I = index.search(np.array([query_embedding]), num_neighbors)

# Print the most similar tweets
print("Here are the most similar tweets to your query:")
for i in I[0]:
    print(all_tweets[i])