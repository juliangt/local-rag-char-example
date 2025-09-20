import os
import pytest
from pathlib import Path
from file_manager import FileManager

@pytest.fixture
def file_manager(tmp_path):
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    return FileManager(docs_path=str(docs_path))

def test_get_all_file_paths_no_files(file_manager, capsys):
    assert file_manager.get_all_file_paths() == []
    captured = capsys.readouterr()
    assert "No supported files found" in captured.out

def test_get_all_file_paths_one_file(file_manager):
    (Path(file_manager.docs_path) / "test.txt").touch()
    expected_path = os.path.join(file_manager.docs_path, "test.txt")
    assert file_manager.get_all_file_paths() == [expected_path]

def test_get_all_file_paths_multiple_files(file_manager):
    (Path(file_manager.docs_path) / "test1.txt").touch()
    (Path(file_manager.docs_path) / "test2.pdf").touch()

    expected_paths = [
        os.path.join(file_manager.docs_path, "test1.txt"),
        os.path.join(file_manager.docs_path, "test2.pdf"),
    ]

    # Use set to ignore order differences in file listing
    assert set(file_manager.get_all_file_paths()) == set(expected_paths)

def test_get_all_file_paths_ignores_unsupported_files(file_manager):
    (Path(file_manager.docs_path) / "test1.txt").touch()
    (Path(file_manager.docs_path) / "unsupported.zip").touch()
    (Path(file_manager.docs_path) / "another.exe").touch()

    expected_path = os.path.join(file_manager.docs_path, "test1.txt")
    assert file_manager.get_all_file_paths() == [expected_path]

def test_get_all_file_paths_returns_full_paths(file_manager):
    (Path(file_manager.docs_path) / "test1.md").touch()

    paths = file_manager.get_all_file_paths()
    assert len(paths) == 1
    assert os.path.isabs(paths[0])
