import os
import argparse
from rag_manager import RAGManager
from interactive_manager import InteractiveManager
from config import config
from file_manager import FileManager

def main():
    os.makedirs(config['index_path'], exist_ok=True)

    parser = argparse.ArgumentParser(description="Chat with one or more documents using RAG.")
    parser.add_argument("file_names", type=str, nargs='*', default=None, help="A list of file names to process.")
    args = parser.parse_args()

    file_manager = FileManager()
    file_paths = file_manager.get_file_paths(args.file_names)

    if not file_paths:
        # FileManager already prints a message if no files are found/selected.
        return

    # Create a unique index name based on the sorted list of file names
    # to ensure that the same set of files gets the same index.
    sorted_files = sorted([os.path.basename(p) for p in file_paths])
    index_name = "_".join(sorted_files) + "_faiss_index"
    index_path = os.path.join(config['index_path'], index_name)

    try:
        rag_manager = RAGManager(file_paths=file_paths, index_path=index_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error during RAG Manager initialization: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    rag_manager.setup()
    
    interactive_manager = InteractiveManager(rag_manager)
    interactive_manager.run()

if __name__ == "__main__":
    main()