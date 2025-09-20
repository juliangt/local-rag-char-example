import os
from config import config

class FileManager:
    def __init__(self, docs_path=None):
        self.docs_path = docs_path or config['docs_path']
        os.makedirs(self.docs_path, exist_ok=True)

    def get_file_paths(self, file_names=None):
        if file_names:
            return self._get_validated_paths(file_names)

        available_files = self._get_available_files()
        if not available_files:
            print(f"No supported files found in the '{self.docs_path}' directory.")
            return None

        return self._choose_files_interactive(available_files)

    def _get_validated_paths(self, file_names):
        validated_paths = []
        for file_name in file_names:
            # Check if it's an absolute path or a path relative to the current directory
            if os.path.exists(file_name):
                validated_paths.append(os.path.abspath(file_name))
                continue

            # Check if it's in the docs_path
            file_path_in_docs = os.path.join(self.docs_path, file_name)
            if os.path.exists(file_path_in_docs):
                validated_paths.append(os.path.abspath(file_path_in_docs))
                continue

            print(f"Error: The file '{file_name}' was not found in the current directory or in '{self.docs_path}'.")
            return None # Exit if any file is not found

        return validated_paths

    def _get_available_files(self):
        supported_extensions = config['supported_extensions']
        return [f for f in os.listdir(self.docs_path) if any(f.endswith(ext) for ext in supported_extensions)]

    def _choose_files_interactive(self, files):
        print("Please choose one or more files to chat with (e.g., 1, 3, 4):")
        sorted_files = sorted(files)
        for i, f in enumerate(sorted_files):
            print(f"[{i+1}] {f}")

        while True:
            try:
                user_input = input("Enter the numbers of the files, separated by commas: ")
                choices = [int(i.strip()) for i in user_input.split(',')]

                if all(1 <= choice <= len(sorted_files) for choice in choices):
                    return [os.path.join(self.docs_path, sorted_files[choice-1]) for choice in choices]
                else:
                    print("Invalid number detected. Please choose from the available file numbers.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
            except Exception as e:
                print(f"An error occurred: {e}")
                return None
