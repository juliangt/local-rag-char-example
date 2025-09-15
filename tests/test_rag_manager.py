import os
import pytest
from unittest.mock import patch, MagicMock
from rag_manager import RAGManager

@pytest.fixture
def rag_manager(tmp_path):
    # Create a dummy file for testing
    dummy_file_path = tmp_path / "dummy.txt"
    dummy_file_path.write_text("This is a dummy file.")

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

def test_init_file_not_found():
    with pytest.raises(FileNotFoundError):
        RAGManager(file_path="non_existent_file.txt", index_path="dummy_index")

def test_init_unsupported_file_type(tmp_path):
    dummy_bad_file_path = tmp_path / "dummy.bad"
    dummy_bad_file_path.write_text("This is a dummy bad file.")
    with pytest.raises(ValueError):
        RAGManager(file_path=str(dummy_bad_file_path), index_path="dummy_index")

def test_init_success(rag_manager):
    assert rag_manager.file_path == str(rag_manager.file_path)
    assert rag_manager.index_path == str(rag_manager.index_path)

@patch('rag_manager.TextLoader')
@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_create_index_txt(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, mock_text_loader, rag_manager):
    mock_ollama_embeddings.return_value = MagicMock()
    mock_chat_ollama.return_value = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance
    mock_text_loader.return_value = MagicMock()

    rag_manager.setup()

    mock_text_loader.assert_called_with(rag_manager.file_path)
    mock_faiss.from_documents.assert_called_once()
    mock_faiss_instance.save_local.assert_called_with(rag_manager.index_path)
    assert rag_manager.chain is not None

@patch('rag_manager.PyPDFLoader')
@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_create_index_pdf(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, mock_pdf_loader, rag_manager, tmp_path):
    dummy_pdf_path = tmp_path / "dummy.pdf"
    dummy_pdf_path.write_text("This is a dummy pdf file.")
    rag_manager.file_path = str(dummy_pdf_path)

    mock_ollama_embeddings.return_value = MagicMock()
    mock_chat_ollama.return_value = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance
    mock_pdf_loader.return_value = MagicMock()

    rag_manager.setup()

    mock_pdf_loader.assert_called_with(rag_manager.file_path)
    mock_faiss.from_documents.assert_called_once()
    mock_faiss_instance.save_local.assert_called_with(rag_manager.index_path)
    assert rag_manager.chain is not None

@patch('rag_manager.UnstructuredMarkdownLoader')
@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_create_index_md(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, mock_md_loader, rag_manager, tmp_path):
    dummy_md_path = tmp_path / "dummy.md"
    dummy_md_path.write_text("# This is a dummy markdown file")
    rag_manager.file_path = str(dummy_md_path)

    mock_ollama_embeddings.return_value = MagicMock()
    mock_chat_ollama.return_value = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance
    mock_md_loader.return_value = MagicMock()

    rag_manager.setup()

    mock_md_loader.assert_called_with(rag_manager.file_path)
    mock_faiss.from_documents.assert_called_once()
    mock_faiss_instance.save_local.assert_called_with(rag_manager.index_path)
    assert rag_manager.chain is not None

@patch('rag_manager.Docx2txtLoader')
@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_create_index_docx(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, mock_docx_loader, rag_manager, tmp_path):
    dummy_docx_path = tmp_path / "dummy.docx"
    dummy_docx_path.write_text("This is a dummy docx file.")
    rag_manager.file_path = str(dummy_docx_path)

    mock_ollama_embeddings.return_value = MagicMock()
    mock_chat_ollama.return_value = MagicMock()
    mock_faiss_instance = MagicMock()
    mock_faiss.from_documents.return_value = mock_faiss_instance
    mock_docx_loader.return_value = MagicMock()

    rag_manager.setup()

    mock_docx_loader.assert_called_with(rag_manager.file_path)
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