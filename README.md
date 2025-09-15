# Local RAG Chat with LangChain and Ollama

This project provides a command-line interface (CLI) to chat with your local text documents. It uses a RAG (Retrieval-Augmented Generation) architecture powered by local models through Ollama and the LangChain framework.

## Features

- **Conversational Chat**: Remembers the context of the conversation for follow-up questions.
- **Local First**: All components, from embeddings to generation models, run locally via Ollama.
- **Multi-Format Support**: Process a variety of common file formats, including:
    - PDF (`.pdf`)
    - Markdown (`.md`)
    - Microsoft Word (`.docx`)
    - Plain Text (`.txt`)
- **External Configuration**: Configure the system using a `config.yaml` file or environment variables.
- **Automatic Indexing**: Automatically creates a vector index of your document on the first run and reuses it in subsequent sessions.
- **Efficient Vector Storage**: Utilizes `FAISS` (Facebook AI Similarity Search) for fast and efficient in-memory vector storage.
- **Simple Directory Structure**: Organizes documents and indexes into dedicated `docs/` and `indexes/` folders.

## Architecture Overview

The script orchestrates the following workflow:

1.  **Indexing (First Run)**: 
    - A text file from the `docs/` folder is loaded.
    - The text is split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
    - Each chunk is converted into a vector embedding using a local Ollama embedding model (e.g., `embeddinggemma`).
    - These embeddings are stored in a `FAISS` vector store, which is saved to the `indexes/` directory.

2.  **Conversational Retrieval**: 
    - When you ask a question, the script first analyzes the conversation history to reformulate your question into a standalone query.
    - This standalone query is used to retrieve the most relevant text chunks from the FAISS vector store.
    - The retrieved chunks, along with the original question and conversation history, are passed to a local Ollama chat model (e.g., `gemma3:270m`).
    - The chat model generates an answer based on the provided context.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **python3 3.9+**
2.  **Ollama**: Make sure the Ollama service is running. You can download it from [ollama.com](https://ollama.com/).
3.  **Ollama Models**: You need to have the specific models pulled. This project defaults to using:
    - `gemma3:270m` for chat.
    - `embeddinggemma` for text embeddings.

    Pull them using the following commands:
    ```bash
    ollama pull gemma3:270m
    ollama pull embeddinggemma
    ```

## Setup & Installation

1.  **Clone the Repository** (or download the files).

2.  **Create `docs` Folder**: Create a directory named `docs` in the project root and place any of the supported files you want to chat with inside it.
    ```bash
    mkdir docs
    mv your_document.pdf docs/
    ```

3.  **Set up Python Environment**: It is highly recommended to use a virtual environment.
    ```bash
    # Create a virtual environment
    python3 -m venv env

    # Activate the environment
    source env/bin/activate
    ```

4.  **Install Dependencies**: Install all the required Python packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Running Tests

To run the unit tests, activate your virtual environment and then execute pytest:

```bash
source env/bin/activate
python3 -m pytest tests/
```

## Usage

To start a chat session, run the `main.py` script from your terminal. You can optionally pass the name of a file in the `docs` directory as an argument. If no filename is provided, the system will attempt to use a default document or prompt you to select one.

```bash
# Start with a specific document
python3 main.py your_document.txt

# Start without specifying a document (e.g., for interactive selection or default document)
python3 main.py
```

- On the first run with a new file, you will see a message indicating that an index is being created. This may take a few moments.
- On subsequent runs, the script will load the existing index, and the session will start much faster.

### Interactive Commands

Once in the chat, you can use the following commands:

- `/clear`: Clears the current index. You will need to run `/reindex` to create a new one.
- `/reindex`: Re-creates the index from the source document.
- `/help`: Shows the list of available commands.
- `/exit`: Exits the chat.

## Configuration

The system can be configured using a `config.yaml` file in the project root or by setting environment variables. The configuration is loaded in the following order of precedence:

1.  **Default values** set in the code.
2.  **`config.yaml` file**.
3.  **Environment variables** (prefixed with `RAG_`).

### `config.yaml`

A `config.yaml` file is created by default with the following values:

```yaml
# Configuration for the RAG system

# Paths
llm_model_path: 'gemma3:270m'
embedding_model_path: 'embeddinggemma'
index_path: './indexes'
docs_path: './docs'

# RAG parameters
chunk_size: 1024
chunk_overlap: 100
k_retriever: 4

# Chat history
replay_history: True
max_replay_history: 5

# LLM parameters
temperature: 0.7
max_new_tokens: 512
n_ctx: 4096
n_gpu_layers: 0
verbose: False
```

### Environment Variables

You can override any of the configuration values by setting environment variables. The environment variable name is the uppercase version of the configuration key, prefixed with `RAG_`.

| Configuration Key | Environment Variable | Default Value |
|---|---|---|
| `llm_model_path` | `RAG_LLM_MODEL_PATH` | `gemma3:270m` |
| `embedding_model_path` | `RAG_EMBEDDING_MODEL_PATH` | `embeddinggemma` |
| `index_path` | `RAG_INDEX_PATH` | `./indexes` |
| `docs_path` | `RAG_DOCS_PATH` | `./docs` |
| `chunk_size` | `RAG_CHUNK_SIZE` | `1024` |
| `chunk_overlap` | `RAG_CHUNK_OVERLAP` | `100` |
| `k_retriever` | `RAG_K_RETRIEVER` | `4` |
| `replay_history` | `RAG_REPLAY_HISTORY` | `True` |
| `max_replay_history` | `RAG_MAX_REPLAY_HISTORY` | `5` |
| `temperature` | `RAG_TEMPERATURE` | `0.7` |
| `max_new_tokens` | `RAG_MAX_NEW_TOKENS` | `512` |
| `n_ctx` | `RAG_N_CTX` | `4096` |
| `n_gpu_layers` | `RAG_N_GPU_LAYERS` | `0` |
| `verbose` | `RAG_VERBOSE` | `False` |

### Exiting the Chat

To end the chat session, type `/exit` and press Enter.
