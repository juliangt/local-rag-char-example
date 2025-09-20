import os
import hashlib
import pickle
from config import config

class CacheManager:
    def __init__(self, cache_path=None):
        self.cache_path = cache_path or config.get('cache_path', './cache')
        os.makedirs(self.cache_path, exist_ok=True)

    def _get_cache_key(self, file_path, embedding_model):
        """Generate a unique cache key based on file content and embedding model."""
        file_hash = self._hash_file_content(file_path)
        return f"{file_hash}_{embedding_model.replace(':', '_')}.pkl"

    def _hash_file_content(self, file_path):
        """Compute the SHA256 hash of the file's content."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get(self, file_path, embedding_model):
        """Load processed documents from the cache."""
        cache_key = self._get_cache_key(file_path, embedding_model)
        cache_file = os.path.join(self.cache_path, cache_key)

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Warning: Could not read cache file {cache_file}. It may be corrupted. Error: {e}")
                return None
        return None

    def set(self, file_path, embedding_model, data):
        """Save processed documents to the cache."""
        cache_key = self._get_cache_key(file_path, embedding_model)
        cache_file = os.path.join(self.cache_path, cache_key)

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except (pickle.PicklingError, OSError) as e:
            print(f"Warning: Could not write to cache file {cache_file}. Error: {e}")
