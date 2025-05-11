from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

def load_chunks(input_file: str):
    """
    Load preprocessed chunks from a JSON file.

    Args:
        input_file (str): Path to the JSON file containing text chunks.

    Returns:
        List[str]: List of text chunks.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        chunks = json.load(file)
    return chunks


def encode_chunks(chunks: list, model_name: str):
    """
    Encode text chunks into vector embeddings using a pre-trained model.

    Args:
        chunks (list): List of text chunks.
        model_name (str): Name of the SentenceTransformers model to use.

    Returns:
        np.ndarray: Array of embeddings.
    """
    #print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    #print("Encoding chunks into embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings, dtype='float32')


def create_faiss_index(embeddings: np.ndarray):
    """
    Create a FAISS index from embeddings.

    Args:
        embeddings (np.ndarray): Array of embeddings.

    Returns:
        faiss.IndexFlatL2: FAISS index.
    """
    dimension = embeddings.shape[1]
    #print(f"Creating FAISS index with dimension: {dimension}")
    
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean distance) index
    index.add(embeddings)  # Add embeddings to the index
    
    #print(f"Added {embeddings.shape[0]} embeddings to the index.")
    return index


def save_faiss_index(index, output_file: str):
    """
    Save the FAISS index to a file.

    Args:
        index: FAISS index.
        output_file (str): Path to save the FAISS index file.
    """
    #print(f"Saving FAISS index to {output_file}...")
    faiss.write_index(index, output_file)

