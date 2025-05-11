import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from typing import List, Dict, Any

def load_faiss_index(index_file: str):
    """
    Load a FAISS index from a file.

    Args:
        index_file (str): Path to the FAISS index file.

    Returns:
        faiss.IndexFlatL2: Loaded FAISS index.
    """
    #print(f"Loading FAISS index from {index_file}...")
    return faiss.read_index(index_file)


def load_text_chunks(input_file: str):
    """
    Load preprocessed text chunks from a JSON file.

    Args:
        input_file (str): Path to the JSON file containing text chunks.

    Returns:
        List[str]: List of text chunks.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        chunks = json.load(file)
    return chunks


def encode_query(query: str, model_name: str) -> np.ndarray:
    """
    Encode a query string into a vector using the specified model.
    
    Args:
        query (str): The query string to encode
        model_name (str): Name of the sentence transformer model to use
        
    Returns:
        np.ndarray: The encoded query vector
    """
    model = SentenceTransformer(model_name)
    return model.encode([query])[0].astype('float32')


def retrieve_similar_chunks(query: str, index: faiss.Index, chunks: List[str], top_k: int, model_name: str = "all-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    """
    Retrieve the most similar chunks for a given query using FAISS.
    
    Args:
        query (str): The query string
        index (faiss.Index): The FAISS index
        chunks (List[str]): List of text chunks
        top_k (int): Number of results to return
        model_name (str): Name of the sentence transformer model to use
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing the retrieved chunks and their scores
    """
    # Encode the query
    query_vector = encode_query(query, model_name)
    
    # Reshape the query vector for FAISS
    query_vector = query_vector.reshape(1, -1)
    
    # Search the index
    distances, indices = index.search(query_vector, top_k)
    
    # Format the results
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(chunks):  # Ensure the index is valid
            results.append({
                "text": chunks[idx],
                "score": float(1 / (1 + distance)),  # Convert distance to similarity score
                "rank": i + 1
            })
    
    return results


