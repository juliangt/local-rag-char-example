# Local RAG Chat with LangChain and Ollama

This project provides a command-line interface (CLI) to chat with your local text documents. It uses a RAG (Retrieval-Augmented Generation) architecture powered by local models through Ollama and the LangChain framework.

## Features

- **Conversational Chat**: Remembers the context of the conversation for follow-up questions.
- **Local First**: All components, from embeddings to generation models, run locally via Ollama.
- **Configurable Models**: Easily switch between different embedding and chat models.
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

1.  **Python 3.9+**
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

2.  **Create `docs` Folder**: Create a directory named `docs` in the project root and place any `.txt` files you want to chat with inside it.
    ```bash
    mkdir docs
    mv your_document.txt docs/
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

## Usage

To start a chat session, run the `main.py` script from your terminal, passing the name of the file in the `docs` directory as an argument.

```bash
python main.py your_document.txt
```

- On the first run with a new file, you will see a message indicating that an index is being created. This may take a few moments.
- On subsequent runs, the script will load the existing index, and the session will start much faster.

### Command-Line Arguments

You can customize the models and index paths using optional arguments:

- `--embedding_model`: Specify the Ollama model to use for embeddings.
  ```bash
  python main.py your_document.txt --embedding_model another-embedding-model
  ```
- `--chat_model`: Specify the Ollama model to use for chat.
  ```bash
  python main.py your_document.txt --chat_model another-chat-model
  ```
- `--index_path`: Specify a custom path to save or load the index.
  ```bash
  python main.py your_document.txt --index_path /custom/path/to/index
  ```

### Exiting the Chat

To end the chat session, type `exit` and press Enter.
