from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Any

def rerank_results(query: str, chunks: List[Dict[str, Any]], model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2") -> List[Dict[str, Any]]:
    """
    Rerank the retrieved chunks using a cross-encoder model.
    
    Args:
        query (str): The user's query
        chunks (List[Dict[str, Any]]): List of retrieved chunks with their scores
        model_name (str): Name of the cross-encoder model to use
        
    Returns:
        List[Dict[str, Any]]: Reranked list of chunks with updated scores
    """
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Prepare the input pairs
    pairs = [(query, chunk["text"]) for chunk in chunks]
    
    # Tokenize the pairs
    features = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        scores = model(**features).logits
    
    # Update the scores in the chunks
    reranked_chunks = []
    for i, chunk in enumerate(chunks):
        reranked_chunks.append({
            "text": chunk["text"],
            "score": float(scores[i][0]),
            "rank": i + 1
        })
    
    # Sort by score in descending order
    reranked_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    return reranked_chunks

