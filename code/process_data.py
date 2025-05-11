from datasets import load_dataset
import json
import time
from huggingface_hub import HfFolder
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_retry_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def preprocess_hf_data(dataset_name: str, output_file: str, chunk_size: int = 300, max_articles: int = 100):
    """
    Load and preprocess a Wikipedia dataset from Hugging Face.

    Args:
        dataset_name (str): The name of the Hugging Face dataset 
        output_file (str): Path to the output JSON file to save chunks.
        chunk_size (int): Maximum length of each chunk (in words).
        max_articles (int): Maximum number of articles to process.

    Returns:
        None
    """
    def chunk_text(text: str, chunk_size: int):
        """
        Split a text into smaller chunks of a specified size.
        """
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Load the dataset with retry logic
    print("Loading the dataset...")
    try:
        # Use a smaller subset of Wikipedia
        dataset = load_dataset(
            dataset_name,
            "20220301.en",
            split="train[:1000]",  # Only load first 1000 articles
            trust_remote_code=True,
            cache_dir="./data/cache"  # Local cache directory
        )
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Retrying with smaller subset...")
        time.sleep(5)  # Wait before retry
        dataset = load_dataset(
            dataset_name,
            "20220301.en",
            split="train[:100]",  # Even smaller subset
            trust_remote_code=True,
            cache_dir="./data/cache"
        )

    # Preprocess articles
    print("Processing articles...")
    chunks = []
    for i, article in enumerate(dataset):
        if i >= max_articles:  # Limit the number of articles processed
            break
        text = article.get("text", "")
        if text:
            text_chunks = chunk_text(text, chunk_size)
            chunks.extend(text_chunks)
        if i % 10 == 0:  # Progress indicator
            print(f"Processed {i} articles...")

    # Save preprocessed chunks to a JSON file
    print(f"Saving {len(chunks)} chunks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print("Preprocessing complete!")

