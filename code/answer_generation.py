from transformers import pipeline
import openai
from typing import List, Dict, Any


def generate_answer(query: str, context: List[Dict[str, Any]], model: str, openai_api_key: str) -> str:
    """
    Generate an answer using OpenAI's GPT model based on the query and context.

    Args:
        query (str): The user's question
        context (List[Dict[str, Any]]): List of relevant context chunks
        model (str): The OpenAI model to use
        openai_api_key (str): OpenAI API key

    Returns:
        str: Generated answer
    """
    # Set up OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Prepare the context
    context_text = "\n\n".join([chunk["text"] for chunk in context])
    
    # Create the prompt
    prompt = f"""Based on the following context, please answer the question. 
    If the answer cannot be found in the context, say "I cannot find a specific answer in the provided context."

    Context:
    {context_text}

    Question: {query}

    Answer:"""

    try:
        # Generate the answer using the new API format
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract the answer from the response
        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return "Sorry, I encountered an error while generating the answer."

