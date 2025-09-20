import os
import shutil
import pytest
from cache_manager import CacheManager

@pytest.fixture
def cache_manager():
    """Fixture to create a CacheManager instance with a temporary cache path."""
    test_cache_path = './test_cache'
    # Setup: create a fresh cache directory
    if os.path.exists(test_cache_path):
        shutil.rmtree(test_cache_path)
    os.makedirs(test_cache_path)

    manager = CacheManager(cache_path=test_cache_path)

    yield manager

    # Teardown: remove the cache directory
    shutil.rmtree(test_cache_path)

@pytest.fixture
def temp_file():
    """Fixture to create a temporary file for testing."""
    file_path = './test_file.txt'
    with open(file_path, 'w') as f:
        f.write('This is a test file.')

    yield file_path

    # Teardown: remove the temporary file
    os.remove(file_path)

def test_cache_manager_init(cache_manager):
    """Test that the CacheManager initializes correctly and creates the cache directory."""
    assert os.path.exists(cache_manager.cache_path)

def test_set_and_get_cache(cache_manager, temp_file):
    """Test setting and getting a cache entry."""
    data_to_cache = {'docs': ['doc1', 'doc2'], 'metadata': 'test'}
    embedding_model = 'test_model'

    # Set data in cache
    cache_manager.set(temp_file, embedding_model, data_to_cache)

    # Get data from cache
    retrieved_data = cache_manager.get(temp_file, embedding_model)

    assert retrieved_data is not None
    assert retrieved_data == data_to_cache

def test_get_non_existent_cache(cache_manager, temp_file):
    """Test getting a non-existent cache entry returns None."""
    retrieved_data = cache_manager.get(temp_file, 'non_existent_model')
    assert retrieved_data is None

def test_cache_invalidation_on_content_change(cache_manager, temp_file):
    """Test that the cache is invalidated if the file content changes."""
    data_to_cache = {'docs': ['doc1']}
    embedding_model = 'test_model'

    # Set initial cache
    cache_manager.set(temp_file, embedding_model, data_to_cache)

    # Modify the file content
    with open(temp_file, 'w') as f:
        f.write('The content has changed.')

    # Try to get the cache with the new content
    retrieved_data = cache_manager.get(temp_file, embedding_model)

    assert retrieved_data is None

def test_cache_invalidation_on_model_change(cache_manager, temp_file):
    """Test that the cache is invalidated if the embedding model changes."""
    data_to_cache = {'docs': ['doc1']}
    embedding_model_1 = 'model_v1'
    embedding_model_2 = 'model_v2'

    # Set cache with the first model
    cache_manager.set(temp_file, embedding_model_1, data_to_cache)

    # Try to get the cache with the second model
    retrieved_data = cache_manager.get(temp_file, embedding_model_2)

    assert retrieved_data is None
