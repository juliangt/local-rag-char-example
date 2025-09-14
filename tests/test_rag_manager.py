import os
import pytest
from unittest.mock import patch, MagicMock
from rag_manager import RAGManager

@pytest.fixture
def rag_manager(tmp_path):
    # Create a dummy file for testing
    dummy_file_path = tmp_path / "dummy.txt"
    dummy_file_path.write_text("This is a dummy file.")

    dummy_pdf_path = tmp_path / "dummy.pdf"
    dummy_pdf_path.write_text("This is a dummy pdf file.")
    
    dummy_index_path = tmp_path / "dummy_index"

    # Mock config
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
    }):
        manager = RAGManager(file_path=str(dummy_file_path), index_path=str(dummy_index_path))
        yield manager
    
    # pytest's tmp_path fixture handles cleanup automatically

def test_init_file_not_found():
    with pytest.raises(FileNotFoundError):
        RAGManager(file_path="non_existent_file.txt", index_path="dummy_index")

def test_init_unsupported_file_type(tmp_path):
    dummy_pdf_path = tmp_path / "dummy.pdf"
    dummy_pdf_path.write_text("This is a dummy pdf file.")
    with pytest.raises(ValueError):
        RAGManager(file_path=str(dummy_pdf_path), index_path="dummy_index")

def test_init_success(rag_manager):
    assert rag_manager.file_path == str(rag_manager.file_path)
    assert rag_manager.index_path == str(rag_manager.index_path)

@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_create_index(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, rag_manager):
    # Mock the language model and embeddings
    mock_ollama_embeddings.return_value = MagicMock()
    mock_chat_ollama.return_value = MagicMock()

    # Mock FAISS
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance
    
    rag_manager.setup()

    # Assert that the index was created and saved
    mock_faiss.from_documents.assert_called_once()
    mock_faiss_instance.save_local.assert_called_with(rag_manager.index_path)
    assert rag_manager.chain is not None

@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_load_index(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, rag_manager):
    # Create a dummy index to simulate loading
    os.makedirs(rag_manager.index_path, exist_ok=True)
    with open(os.path.join(rag_manager.index_path, "dummy.faiss"), "w") as f:
        f.write("dummy index data")

    # Mock the language model and embeddings
    mock_ollama_embeddings.return_value = MagicMock()
    mock_chat_ollama.return_value = MagicMock()

    # Mock FAISS
    mock_faiss_instance = MagicMock()
    mock_faiss.load_local.return_value = mock_faiss_instance

    rag_manager.setup()

    # Assert that the index was loaded
    mock_faiss.load_local.assert_called_with(rag_manager.index_path, mock_ollama_embeddings.return_value, allow_dangerous_deserialization=True)
    assert rag_manager.chain is not None

@patch('rag_manager.RAGManager.setup')
def test_ask(mock_setup, rag_manager):
    rag_manager.chain = MagicMock()
    rag_manager.chain.invoke.return_value = {"answer": "This is a test answer."}
    
    answer = rag_manager.ask("What is the meaning of life?", [])
    
    assert answer == "This is a test answer."
    rag_manager.chain.invoke.assert_called_once()