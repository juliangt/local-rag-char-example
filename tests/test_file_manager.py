import os
import pytest
from unittest.mock import patch
from pathlib import Path
from file_manager import FileManager
from config import config

@pytest.fixture
def file_manager(tmp_path):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    # Create some dummy files
    (docs_path / "test1.txt").touch()
    (docs_path / "test2.pdf").touch()
    (docs_path / "unsupported.zip").touch()
    return FileManager(docs_path=str(docs_path))

def test_get_file_paths_with_valid_names(file_manager):
    paths = file_manager.get_file_paths(["test1.txt", "test2.pdf"])
    assert len(paths) == 2
    assert set(os.path.basename(p) for p in paths) == {"test1.txt", "test2.pdf"}

def test_get_file_paths_with_invalid_name(file_manager, capsys):
    paths = file_manager.get_file_paths(["test1.txt", "nonexistent.txt"])
    assert paths is None
    captured = capsys.readouterr()
    assert "Error: The file 'nonexistent.txt' was not found" in captured.out

def test_get_file_paths_with_absolute_path(file_manager, tmp_path):
    # Create a file outside the docs directory
    abs_file = tmp_path / "absolute.txt"
    abs_file.touch()
    paths = file_manager.get_file_paths([str(abs_file)])
    assert len(paths) == 1
    assert paths[0] == str(abs_file)

@patch('builtins.input', return_value='1')
def test_choose_files_interactive_single(mock_input, file_manager):
    paths = file_manager.get_file_paths()
    assert len(paths) == 1
    assert os.path.basename(paths[0]) == "test1.txt"

@patch('builtins.input', return_value='1, 2')
def test_choose_files_interactive_multiple(mock_input, file_manager):
    paths = file_manager.get_file_paths()
    assert len(paths) == 2
    assert set(os.path.basename(p) for p in paths) == {"test1.txt", "test2.pdf"}

@patch('builtins.input', side_effect=['invalid', '1'])
def test_choose_files_interactive_invalid_then_valid(mock_input, file_manager, capsys):
    paths = file_manager.get_file_paths()
    assert len(paths) == 1
    assert os.path.basename(paths[0]) == "test1.txt"
    captured = capsys.readouterr()
    assert "Invalid input. Please enter numbers separated by commas." in captured.out

def test_no_supported_files_found(tmp_path, capsys):
    # Create a file manager with an empty docs directory
    empty_docs_path = tmp_path / "empty_docs"
    empty_docs_path.mkdir()
    fm = FileManager(docs_path=str(empty_docs_path))

    assert fm.get_file_paths() is None
    captured = capsys.readouterr()
    assert "No supported files found" in captured.out

def test_available_files_ignores_unsupported(file_manager):
    # The fixture already creates an unsupported file
    available = file_manager._get_available_files()
    assert len(available) == 2
    assert "unsupported.zip" not in available
