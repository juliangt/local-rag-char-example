import os
from config import config

class FileManager:
    def __init__(self, docs_path=None):
        self.docs_path = docs_path or config['docs_path']
        os.makedirs(self.docs_path, exist_ok=True)

    def get_all_file_paths(self):
        """
        Returns a list of full paths to all supported files in the docs directory.
        """
        supported_extensions = ['.txt', '.pdf', '.md', '.docx']
        file_paths = []
        for f in os.listdir(self.docs_path):
            if any(f.endswith(ext) for ext in supported_extensions):
                file_paths.append(os.path.join(self.docs_path, f))

        if not file_paths:
            print(f"No supported files found in the '{self.docs_path}' directory.")

        return file_paths
