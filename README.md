# RAG-Whisper

A powerful FAQ generation system that uses Retrieval-Augmented Generation (RAG) to create context-aware answers from a knowledge base. This project combines state-of-the-art language models with efficient document retrieval to provide accurate and relevant answers to user queries.

## Author
- [H-ario-m](https://github.com/H-ario-m)

## Features

- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with language model generation for more accurate answers
- **FAISS Vector Search**: Efficient similarity search for retrieving relevant context
- **Cross-Encoder Reranking**: Improves answer relevance through semantic reranking
- **OpenAI Integration**: Uses GPT models for high-quality answer generation
- **Wikipedia Knowledge Base**: Default dataset with option to use custom data
- **Configurable Parameters**: Easy customization of chunk size, model selection, and more

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Sufficient disk space for the knowledge base (default: Wikipedia dataset)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/H-ario-m/RAG-Wishper.git
cd RAG-Whisper
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Configuration

The system can be configured through `config.json`:

```json
{
    "dataset_name": "wikipedia",
    "chunks_file": "./data/wikipedia_chunks.json",
    "faiss_file": "./data/document_index.faiss",
    "chunk_size": 300,
    "max_articles": 100,
    "top_k_results": 5,
    "encoding_model": "all-MiniLM-L6-v2",
    "reranking_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "answer_generation_model": "gpt-3.5-turbo"
}
```

### Configuration Options

- `dataset_name`: Name of the dataset to use (default: "wikipedia")
- `chunks_file`: Path to save processed text chunks
- `faiss_file`: Path to save the FAISS index
- `chunk_size`: Maximum size of text chunks in words
- `max_articles`: Maximum number of articles to process
- `top_k_results`: Number of top results to retrieve
- `encoding_model`: Model for encoding text into vectors
- `reranking_model`: Model for reranking retrieved results
- `answer_generation_model`: OpenAI model for generating answers

## Usage

1. Run the main script:
```bash
python main.py
```

2. The script will:
   - Process and index the data if not already done
   - Create a FAISS index for efficient retrieval
   - Start an interactive query session

3. Enter your questions when prompted. Type 'Q' to quit.

## Project Structure

```
RAG-Whisper/
├── code/
│   ├── answer_generation.py  # Answer generation using OpenAI
│   ├── create_indexing.py    # FAISS index creation
│   ├── process_data.py       # Data preprocessing
│   ├── retrieval.py          # Document retrieval
│   └── reranking.py          # Result reranking
├── data/                     # Processed data and indices
├── .env                      # Environment variables
├── config.json              # Configuration file
├── main.py                  # Main script
└── requirements.txt         # Python dependencies
```

## How It Works

1. **Data Processing**:
   - Loads and chunks the knowledge base (default: Wikipedia)
   - Creates embeddings using sentence transformers
   - Builds a FAISS index for efficient retrieval
     ![Screenshot 2025-05-11 163713](https://github.com/user-attachments/assets/e0999898-be06-4d35-a5a0-8b44e55c9d8c)


2. **Query Processing**:
   - Encodes the user query into a vector
   - Retrieves similar chunks using FAISS
   - Reranks results using a cross-encoder
   - Generates an answer using GPT

3. **Answer Generation**:
   - Combines retrieved context with the query
   - Uses GPT to generate a coherent answer
   - Returns the final response to the user
