import os
import pytest
from unittest.mock import patch, MagicMock
from rag_manager import RAGManager

@pytest.fixture
def rag_manager(tmp_path):
    # Create a dummy docs directory and a dummy file for testing
    dummy_docs_path = tmp_path / "docs"
    dummy_docs_path.mkdir()
    dummy_file_path = dummy_docs_path / "dummy.txt"
    dummy_file_path.write_text("This is a dummy file.")

    dummy_index_root_path = tmp_path / "indexes"
    dummy_index_root_path.mkdir()

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
        'verbose': False,
        'docs_path': str(dummy_docs_path),
        'index_path': str(dummy_index_root_path)
    }):
        manager = RAGManager()
        yield manager

def test_init_success(rag_manager):
    from rag_manager import config # Import config inside the test to get the patched version
    assert rag_manager.active_document == "dummy.txt"
    assert rag_manager.file_path == os.path.join(config['docs_path'], "dummy.txt")
    assert rag_manager.index_path == os.path.join(config['index_path'], "dummy_faiss_index")
    assert "dummy.txt" in rag_manager.documents

@patch('rag_manager.RAGManager._create_index')
@patch('rag_manager.RAGManager._load_index')
@patch('rag_manager.TextLoader')
@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_create_index_txt(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, mock_text_loader, mock_load_index, mock_create_index, rag_manager, tmp_path):
    from rag_manager import config # Import config inside the test to get the patched version
    
    # Create a new dummy text file for this test
    new_dummy_txt_path = tmp_path / "docs" / "new_dummy.txt"
    new_dummy_txt_path.write_text("This is a new dummy text file.")
    
    # Add it to the discovered documents
    rag_manager.documents.append("new_dummy.txt")
    rag_manager.documents.sort() # Keep it sorted

    # Ensure _create_index is called when setup is invoked
    mock_create_index.return_value = None # _create_index doesn't return anything
    mock_load_index.return_value = None # _load_index doesn't return anything

    rag_manager.switch_document("new_dummy.txt")

    # Now call setup, which should trigger _create_index
    rag_manager.setup()

    mock_create_index.assert_called_once()
    mock_load_index.assert_not_called()
    assert rag_manager.chain is not None # Chain should still be set up by setup()

@patch('rag_manager.RAGManager._create_index')
@patch('rag_manager.RAGManager._load_index')
@patch('rag_manager.PyPDFLoader')
@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_create_index_pdf(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, mock_pdf_loader, mock_load_index, mock_create_index, rag_manager, tmp_path):
    from rag_manager import config # Import config inside the test to get the patched version

    # Create a new dummy PDF file for this test
    new_dummy_pdf_path = tmp_path / "docs" / "new_dummy.pdf"
    new_dummy_pdf_path.write_text("This is a new dummy PDF file.")

    # Add it to the discovered documents
    rag_manager.documents.append("new_dummy.pdf")
    rag_manager.documents.sort() # Keep it sorted

    # Ensure _create_index is called when setup is invoked
    mock_create_index.return_value = None
    mock_load_index.return_value = None

    rag_manager.switch_document("new_dummy.pdf")

    rag_manager.setup()

    mock_create_index.assert_called_once()
    mock_load_index.assert_not_called()
    assert rag_manager.chain is not None

@patch('rag_manager.RAGManager._create_index')
@patch('rag_manager.RAGManager._load_index')
@patch('rag_manager.UnstructuredMarkdownLoader')
@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_create_index_md(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, mock_md_loader, mock_load_index, mock_create_index, rag_manager, tmp_path):
    from rag_manager import config # Import config inside the test to get the patched version

    # Create a new dummy Markdown file for this test
    new_dummy_md_path = tmp_path / "docs" / "new_dummy.md"
    new_dummy_md_path.write_text("# This is a new dummy markdown file")

    # Add it to the discovered documents
    rag_manager.documents.append("new_dummy.md")
    rag_manager.documents.sort() # Keep it sorted

    # Ensure _create_index is called when setup is invoked
    mock_create_index.return_value = None
    mock_load_index.return_value = None

    rag_manager.switch_document("new_dummy.md")

    rag_manager.setup()

    mock_create_index.assert_called_once()
    mock_load_index.assert_not_called()
    assert rag_manager.chain is not None

@patch('rag_manager.RAGManager._create_index')
@patch('rag_manager.RAGManager._load_index')
@patch('rag_manager.Docx2txtLoader')
@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_create_index_docx(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, mock_docx_loader, mock_load_index, mock_create_index, rag_manager, tmp_path):
    from rag_manager import config # Import config inside the test to get the patched version

    # Create a new dummy DOCX file for this test
    new_dummy_docx_path = tmp_path / "docs" / "new_dummy.docx"
    new_dummy_docx_path.write_text("This is a new dummy DOCX file.")

    # Add it to the discovered documents
    rag_manager.documents.append("new_dummy.docx")
    rag_manager.documents.sort() # Keep it sorted

    # Ensure _create_index is called when setup is invoked
    mock_create_index.return_value = None
    mock_load_index.return_value = None

    rag_manager.switch_document("new_dummy.docx")

    rag_manager.setup()

    mock_create_index.assert_called_once()
    mock_load_index.assert_not_called()
    assert rag_manager.chain is not None

@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_load_index(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, rag_manager, tmp_path):
    from rag_manager import config # Import config inside the test to get the patched version

    # Create a dummy index directory for the default document
    dummy_index_path = tmp_path / "indexes" / "dummy_faiss_index"
    os.makedirs(dummy_index_path, exist_ok=True)
    with open(os.path.join(dummy_index_path, "dummy.faiss"), "w") as f:
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

def test_list_documents(rag_manager):
    # The fixture already creates 'dummy.txt'
    assert "dummy.txt" in rag_manager.list_documents()
    assert len(rag_manager.list_documents()) >= 1 # At least dummy.txt

def test_switch_document(rag_manager, tmp_path):
    from rag_manager import config # Import config inside the test to get the patched version

    # Create another dummy document
    new_doc_path = tmp_path / "docs" / "another.txt"
    new_doc_path.write_text("This is another document.")
    
    # Add it to the discovered documents
    rag_manager.documents.append("another.txt")
    rag_manager.documents.sort()

    # Switch to the new document
    rag_manager.switch_document("another.txt")

    assert rag_manager.active_document == "another.txt"
    assert rag_manager.file_path == os.path.join(config['docs_path'], "another.txt")
    assert rag_manager.index_path == os.path.join(config['index_path'], "another_faiss_index")
    # setup() should have been called by switch_document
    # We can't directly assert setup() was called without patching it,
    # but we can check the resulting state.


@patch('rag_manager.OllamaEmbeddings')
@patch('rag_manager.FAISS')
@patch('rag_manager.ChatOllama')
def test_setup_load_index(mock_chat_ollama, mock_faiss, mock_ollama_embeddings, rag_manager, tmp_path):
    from rag_manager import config # Import config inside the test to get the patched version

    # Create a dummy index directory for the default document
    dummy_index_path = tmp_path / "indexes" / "dummy_faiss_index"
    os.makedirs(dummy_index_path, exist_ok=True)
    with open(os.path.join(dummy_index_path, "dummy.faiss"), "w") as f:
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

def test_list_documents(rag_manager):
    # The fixture already creates 'dummy.txt'
    assert "dummy.txt" in rag_manager.list_documents()
    assert len(rag_manager.list_documents()) >= 1 # At least dummy.txt

def test_switch_document(rag_manager, tmp_path):
    from rag_manager import config # Import config inside the test to get the patched version

    # Create another dummy document
    new_doc_path = tmp_path / "docs" / "another.txt"
    new_doc_path.write_text("This is another document.")
    
    # Add it to the discovered documents
    rag_manager.documents.append("another.txt")
    rag_manager.documents.sort()

    # Switch to the new document
    rag_manager.switch_document("another.txt")

    assert rag_manager.active_document == "another.txt"
    assert rag_manager.file_path == os.path.join(config['docs_path'], "another.txt")
    assert rag_manager.index_path == os.path.join(config['index_path'], "another_faiss_index")
    # setup() should have been called by switch_document
    # We can't directly assert setup() was called without patching it,
    # but we can check the resulting state.