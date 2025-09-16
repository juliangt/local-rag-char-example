import pytest
from unittest.mock import MagicMock, patch
from interactive_manager import InteractiveManager
from config import config

@pytest.fixture
def interactive_manager():
    rag_manager = MagicMock()
    rag_manager.active_document = "dummy.txt" # Default active document
    rag_manager.documents = ["dummy.txt", "another.txt"] # Discovered documents
    rag_manager.list_documents.return_value = rag_manager.documents
    rag_manager.index_path = "dummy_index_path" # For clear_index
    rag_manager.file_path = "dummy_file_path" # For welcome message
    return InteractiveManager(rag_manager)

@patch('interactive_manager.InteractiveManager.clear_index')
@patch('builtins.input', side_effect=['/clear', '/exit'])
def test_run_clear_command(mock_input, mock_clear_index, interactive_manager, capsys):
    with pytest.raises(SystemExit) as e:
        interactive_manager.run()
    assert e.type == SystemExit

    mock_clear_index.assert_called_once()

@patch('interactive_manager.InteractiveManager.reindex')
@patch('builtins.input', side_effect=['/reindex', '/exit'])
def test_run_reindex_command(mock_input, mock_reindex, interactive_manager):
    with pytest.raises(SystemExit) as e:
        interactive_manager.run()
    assert e.type == SystemExit

    mock_reindex.assert_called_once()

@patch('interactive_manager.InteractiveManager.list_docs')
@patch('builtins.input', side_effect=['/list_docs', '/exit'])
def test_run_list_docs_command(mock_input, mock_list_docs, interactive_manager):
    with pytest.raises(SystemExit) as e:
        interactive_manager.run()
    assert e.type == SystemExit

    mock_list_docs.assert_called_once()

@patch('interactive_manager.InteractiveManager.use_doc')
@patch('builtins.input', side_effect=['/use_doc another.txt', '/exit'])
def test_run_use_doc_command(mock_input, mock_use_doc, interactive_manager):
    with pytest.raises(SystemExit) as e:
        interactive_manager.run()
    assert e.type == SystemExit

    mock_use_doc.assert_called_once_with(['another.txt'])

@patch('builtins.input', side_effect=['/help', '/exit'])
def test_run_help_command(mock_input, interactive_manager, capsys):
    with pytest.raises(SystemExit) as e:
        interactive_manager.run()
    assert e.type == SystemExit

    captured = capsys.readouterr()
    assert "Available commands:" in captured.out

@patch('builtins.input', side_effect=['/unknown', '/exit'])
def test_run_unknown_command(mock_input, interactive_manager, capsys):
    with pytest.raises(SystemExit) as e:
        interactive_manager.run()
    assert e.type == SystemExit

    captured = capsys.readouterr()
    assert "Unrecognized command: /unknown" in captured.out

@patch('builtins.input', side_effect=['hello', '/exit'])
def test_run_user_query(mock_input, interactive_manager):
    with patch.dict(config, {'replay_history': False}):
        interactive_manager.rag_manager.ask.return_value = "world"
        with pytest.raises(SystemExit) as e:
            interactive_manager.run()
        assert e.type == SystemExit

        interactive_manager.rag_manager.ask.assert_called_once_with('hello', [])

@patch('builtins.input', side_effect=['/exit'])
def test_run_exit_command(mock_input, interactive_manager):
    with pytest.raises(SystemExit) as e:
        interactive_manager.run()
    assert e.type == SystemExit
