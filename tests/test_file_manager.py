import os
import pytest
from unittest.mock import patch
from pathlib import Path
from file_manager import FileManager

@pytest.fixture
def file_manager(tmp_path):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return FileManager(docs_path=str(docs_path))

def test_get_file_path_no_files(file_manager, capsys):
    assert file_manager.get_file_path() is None
    captured = capsys.readouterr()
    assert "No supported files found" in captured.out

def test_get_file_path_one_file(file_manager):
    (Path(file_manager.docs_path) / "test.txt").touch()
    assert file_manager.get_file_path() == os.path.join(file_manager.docs_path, "test.txt")

@patch('builtins.input', return_value='1')
def test_get_file_path_multiple_files(mock_input, file_manager):
    (Path(file_manager.docs_path) / "test1.txt").touch()
    (Path(file_manager.docs_path) / "test2.txt").touch()
    assert file_manager.get_file_path() == os.path.join(file_manager.docs_path, "test1.txt")

def test_get_file_path_with_file_name_exists_current_dir(file_manager, tmp_path):
    (tmp_path / "test.txt").touch()
    os.chdir(tmp_path)
    assert file_manager.get_file_path("test.txt") == os.path.abspath("test.txt")

def test_get_file_path_with_file_name_exists_docs_dir(file_manager):
    (Path(file_manager.docs_path) / "test.txt").touch()
    assert os.path.basename(file_manager.get_file_path("test.txt")) == "test.txt"

def test_get_file_path_with_file_name_not_exists(file_manager, capsys):
    assert file_manager.get_file_path("non_existent_file.txt") is None
    captured = capsys.readouterr()
    assert "Error: The file 'non_existent_file.txt' was not found" in captured.out
