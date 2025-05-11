# Context-Aware FAQ Generator with RAG

import json
import os
import sys
import faiss
import numpy as np
import shutil
import openai

sys.path.append('./code')
from answer_generation import *
from create_indexing import *
from process_data import *
from retrieval import *
from reranking import *

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
open_ai_key = os.getenv("OPENAI_API_KEY")

# Verify API key
if not open_ai_key:
    print("Error: OPENAI_API_KEY not found in .env file")
    sys.exit(1)

# Test the API key
try:
    client = openai.OpenAI(api_key=open_ai_key)
    # Make a simple API call to verify the key
    client.models.list()
    print("API key verified successfully!")
except Exception as e:
    print(f"Error verifying API key: {str(e)}")
    print("Please check your API key and billing status at https://platform.openai.com/account/billing")
    sys.exit(1)

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Access variables from the config
dataset_name = config.get("dataset_name", "wikipedia")
chunks_file = config.get("output_file", "./data/wikipedia_chunks.json")
chunk_size = config.get("chunk_size", 300)
max_articles = config.get("max_articles", 1000)
faiss_file = config.get("faiss_file", "./data/document_index.faiss")
top_k_results = config.get("top_k_results", 5)

encoding_model = config.get("encoding_model", "all-MiniLM-L6-v2")
reranking_model = config.get("reranking_model", "cross-encoder/ms-marco-MiniLM-L-12-v2")
answer_generation_model = config.get("answer_generation_model", "EleutherAI/gpt-neo-1.3B")

# Create data directory if it doesn't exist
if not os.path.exists('./data'):
    os.makedirs('./data')

def process_data():
    """Process and index the data."""
    print("Processing Data...")
    
    # Preprocess and encode text data
    preprocess_hf_data(
        dataset_name=config["dataset_name"],
        output_file=config["chunks_file"],
        chunk_size=config["chunk_size"],
        max_articles=config["max_articles"]
    )

    # Load and encode text chunks
    text_chunks = load_chunks(config["chunks_file"])
    embeddings = encode_chunks(text_chunks, config["encoding_model"])

    # Create and save FAISS index
    faiss_index = create_faiss_index(embeddings)
    save_faiss_index(faiss_index, config["faiss_file"])

    print("Indexing complete!\n\n")

def process_query(query: str) -> str:
    """
    Process a user query through the RAG pipeline.
    """
    try:
        # Load the FAISS index
        index = faiss.read_index(config["faiss_file"])
        
        # Load the chunks
        with open(config["chunks_file"], 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Get similar chunks
        similar_chunks = retrieve_similar_chunks(
            query=query,
            index=index,
            chunks=chunks,
            top_k=config["top_k_results"],
            model_name=config["encoding_model"]
        )
        
        # Rerank the results
        reranked_results = rerank_results(
            query=query,
            chunks=similar_chunks,
            model_name=config["reranking_model"]
        )
        
        # Generate the final answer
        final_answer = generate_answer(
            query=query,
            context=reranked_results,
            model=config["answer_generation_model"],
            openai_api_key=open_ai_key
        )
        
        return final_answer
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        if "insufficient_quota" in str(e):
            print("\nYour OpenAI API key has insufficient quota. Please:")
            print("1. Check your billing status at https://platform.openai.com/account/billing")
            print("2. Add payment method if needed")
            print("3. Try a different API key")
        return "Sorry, I encountered an error while processing your query."

if __name__ == "__main__":
    # Check if we need to process data
    if not os.path.exists('./data') or not os.listdir('./data'):
        process_data()
    else:
        answer = input("It seems that your data directory is not empty. Would you like to clear it? (Y/n): ")
        if answer.lower() == 'y':
            try:
                # Remove the entire data directory and recreate it
                shutil.rmtree('./data')
                os.makedirs('./data')
                print("Data directory cleared.")
            except Exception as e:
                print(f"Error clearing data directory: {str(e)}")
                print("Please try closing any applications that might be using the files and try again.")
                sys.exit(1)
            process_data()

    # Process queries
    while True:
        user_query = input("\nEnter your Topic (Q to Quit): ")
        if user_query.lower() == 'q':
            break
            
        final_answer = process_query(user_query)
        print("\nAnswer:", final_answer, "\n")

    