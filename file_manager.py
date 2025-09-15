import os
from config import config

class FileManager:
    def __init__(self, docs_path=None):
        self.docs_path = docs_path or config['docs_path']
        os.makedirs(self.docs_path, exist_ok=True)

    def get_file_path(self, file_name=None):
        if file_name:
            if os.path.exists(file_name):
                return os.path.abspath(file_name)
            else:
                file_path = os.path.join(self.docs_path, file_name)
                if os.path.exists(file_path):
                    return os.path.abspath(file_path)
                else:
                    print(f"Error: The file '{file_name}' was not found in the current directory or in the '{self.docs_path}' directory.")
                    return None

        files = self._get_available_files()
        if not files:
            print(f"No supported files found in the '{self.docs_path}' directory.")
            return None
        elif len(files) == 1:
            return os.path.join(self.docs_path, files[0])
        else:
            return self._choose_file(files)

    def _get_available_files(self):
        supported_extensions = ['.txt', '.pdf', '.md', '.docx']
        return [f for f in os.listdir(self.docs_path) if any(f.endswith(ext) for ext in supported_extensions)]

    def _choose_file(self, files):
        print("Please choose a file to chat with:")
        for i, f in enumerate(files):
            print(f"[{i+1}] {f}")
        while True:
            try:
                choice = int(input("Enter the number of the file: "))
                if 1 <= choice <= len(files):
                    return os.path.join(self.docs_path, files[choice-1])
                else:
                    print("Invalid number.")
            except ValueError:
                print("Please enter a number.")
