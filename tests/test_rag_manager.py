import os
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from rag_manager import RAGManager
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader

@pytest.fixture
def mock_config():
    with patch('rag_manager.config', {
        'embedding_model_path': 'dummy_embedding_model',
        'llm_model_path': 'dummy_chat_model',
        'chunk_size': 100,
        'chunk_overlap': 10,
        'k_retriever': 3,
        'temperature': 0.7,
        'max_new_tokens': 512,
        'n_ctx': 4096,
        'n_gpu_layers': 0,
        'verbose': False
    }) as mock_config:
        yield mock_config

@pytest.fixture
def rag_manager(tmp_path, mock_config):
    # Create dummy files for testing
    dummy_file_1_path = tmp_path / "dummy1.txt"
    dummy_file_1_path.write_text("This is the first dummy file.")
    dummy_file_2_path = tmp_path / "dummy2.md"
    dummy_file_2_path.write_text("# This is a markdown file.")

    file_paths = [str(dummy_file_1_path), str(dummy_file_2_path)]
    dummy_index_path = tmp_path / "multi_doc_index"

    manager = RAGManager(file_paths=file_paths, index_path=str(dummy_index_path))
    return manager

def test_init_no_files_provided(tmp_path):
    with pytest.raises(FileNotFoundError, match="No files provided to process."):
        RAGManager(file_paths=[], index_path=str(tmp_path / "some_index"))

def test_init_file_not_found(tmp_path):
    existing_file = tmp_path / "exists.txt"
    existing_file.write_text("I exist.")
    with pytest.raises(FileNotFoundError):
        RAGManager(file_paths=[str(existing_file), "non_existent_file.txt"], index_path="dummy_index")

def test_init_unsupported_file_type(tmp_path):
    dummy_good_file = tmp_path / "dummy.txt"
    dummy_good_file.write_text("This is a dummy good file.")
    dummy_bad_file = tmp_path / "dummy.bad"
    dummy_bad_file.write_text("This is a dummy bad file.")
    with pytest.raises(ValueError):
        RAGManager(file_paths=[str(dummy_good_file), str(dummy_bad_file)], index_path="dummy_index")

def test_init_success(rag_manager):
    assert len(rag_manager.file_paths) == 2
    assert "dummy1.txt" in rag_manager.file_paths[0]
    assert "dummy2.md" in rag_manager.file_paths[1]

@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
@patch('rag_manager.TextLoader')
@patch('rag_manager.UnstructuredMarkdownLoader')
def test_setup_creates_combined_index(mock_md_loader, mock_txt_loader, mock_chat_ollama, mock_faiss, mock_ollama_embeddings, rag_manager):
    # Mock loaders to return dummy documents
    mock_doc_txt = MagicMock()
    mock_doc_txt.page_content = "text content"
    mock_doc_txt.metadata = {"source": "dummy1.txt"}
    mock_doc_md = MagicMock()
    mock_doc_md.page_content = "markdown content"
    mock_doc_md.metadata = {"source": "dummy2.md"}
    mock_txt_loader.return_value.load.return_value = [mock_doc_txt]
    mock_md_loader.return_value.load.return_value = [mock_doc_md]

    mock_ollama_embeddings.return_value = MagicMock()
    mock_chat_ollama.return_value = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance

    rag_manager.setup()

    # Check that the correct loaders were called for each file
    mock_txt_loader.assert_called_with(rag_manager.file_paths[0])
    mock_md_loader.assert_called_with(rag_manager.file_paths[1])

    # Check that FAISS was created with documents from all loaders
    docs_passed_to_faiss = mock_faiss.from_documents.call_args[0][0]
    assert len(docs_passed_to_faiss) == 2
    page_contents = {doc.page_content for doc in docs_passed_to_faiss}
    assert {"text content", "markdown content"} == page_contents

    # Check that the combined index is saved
    mock_faiss_instance.save_local.assert_called_with(rag_manager.index_path)
    assert rag_manager.chain is not None

@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_loads_existing_index(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, rag_manager):
    # Simulate an existing index
    os.makedirs(rag_manager.index_path, exist_ok=True)
    # Create a dummy file to make the directory non-empty
    (Path(rag_manager.index_path) / "index.faiss").touch()

    mock_ollama_embeddings.return_value = MagicMock()
    mock_chat_ollama.return_value = MagicMock()
    mock_faiss.load_local.return_value = MagicMock()

    rag_manager.setup()

    mock_faiss.load_local.assert_called_with(rag_manager.index_path, mock_ollama_embeddings.return_value, allow_dangerous_deserialization=True)
    assert rag_manager.chain is not None

@patch('rag_manager.RAGManager.setup')
def test_ask_functionality(mock_setup, rag_manager):
    rag_manager.chain = MagicMock()
    rag_manager.chain.invoke.return_value = {"answer": "This is a test answer."}
    
    answer = rag_manager.ask("What is the meaning of life?", [])
    
    assert answer == "This is a test answer."
    rag_manager.chain.invoke.assert_called_once()