# Mycelium Project Chatbot

## Project Overview

This NLP chatbot is designed to assist users in exploring and understanding the Mycelium Project, a scientific and applied research initiative by students at InHolland University of Applied Sciences in Haarlem. The project investigates innovative uses of fungal mycelium in domains such as bioremediation, sustainable materials, and environmental monitoring. The chatbot leverages a combination of vector-based retrieval and local LLM inference to provide contextually grounded responses based on a curated corpus.

## Installation Instructions

Assuming you have already cloned the repository:

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install and set up Ollama:**

   This project requires a local instance of Ollama to run the `llama3.2` model.

   - [Install Ollama](https://ollama.com/)

   - Pull the required model:

     ```bash
     ollama pull llama3:2
     ```

   - Start the Ollama server:

     ```bash
     ollama serve
     ```

   > ⚠️ **Important:** If Ollama is not running or the `llama3.2` model is not available, the program will raise warnings and responses from the LLM will fail.

4. **Prepare data folders:**

   When you run the app, the following directories will be created automatically if they do not exist:

   - `./data/raw/` - for source documents (supports `.txt` and `.docx`)
   - `./data/vector_store/` - stores the TF-IDF vector index

   If `./data/raw/` is empty, a warning will be shown and no document-based context will be available.

## Usage Guide

To start the chatbot:

```bash
python app.py
```

You can interact with the assistant by typing your question. To exit the session, type:

```text
stop
exit
quit
```

## Architecture Description

- **app.py**: Entry point that ensures directory setup and launches the chatbot loop.
- **chat.py**: Core logic for context retrieval, question-answering, and interaction with the local LLM (via Ollama's OpenAI-compatible API).
- **vector_db.py**: Manages document vectorization, persistent storage, and similarity search.
- **embeddings.py**: Wraps a TF-IDF vectorizer for simple semantic representation.
- **preprocessing.py**: Cleans and chunks `.txt` and `.docx` files for embedding.
- **utils.py**: Some general functions to check with the Ollama server

**Data Flow:**

1. Raw documents are cleaned and chunked.
2. Chunks are embedded using TF-IDF.
3. Vectors are stored with the corresponding text.
4. User query is embedded and matched against stored vectors.
5. The top-matching snippets are used to inform the LLM response.
6. If context is insufficient, an LLM-driven relevance checker rephrases and retries the query.

## NLP Approach Explanation

This project employs a **retrieval-augmented generation (RAG)** architecture:

- **Vectorization**: Uses TF-IDF for fast, interpretable embedding of document chunks.
- **Similarity Search**: Cosine similarity ranks document chunks relevant to the user's query.
- **Relevance Checking**: A local LLM (via Ollama) determines if the retrieved context is sufficient to answer the query, and iteratively improves the query if necessary.
- **Response Generation**: Combines a system prompt with relevant context and the user query to produce a reply using the `llama3.2` model.

The chatbot adapts its language based on the input (Dutch or English) and prioritizes factual, context-driven answers.

## Known Limitations

- **Empty `raw/` folder**: No contextual data will be available if the folder is empty.
- **Static TF-IDF Embeddings**: Semantic matching is limited by TF-IDF's vocabulary and shallow context representation.
- **Relevance Loop Overhead**: The LLM-driven `_contextCheck()` may slow down responses if context is deemed insufficient.
- **No Web UI**: The system runs in a terminal interface only.
- **Limited File Support**: Only `.docx` and `.txt` documents are supported.
