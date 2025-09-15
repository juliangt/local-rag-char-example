import os
import argparse
from rag_manager import RAGManager
from interactive_manager import InteractiveManager
from config import config
from file_manager import FileManager

def main():
    os.makedirs(config['index_path'], exist_ok=True)

    parser = argparse.ArgumentParser(description="Chat with a document using RAG.")
    parser.add_argument("file_name", type=str, nargs='?', default=None, help="Name or path of the file.")
    args = parser.parse_args()

    file_manager = FileManager()
    file_path = file_manager.get_file_path(args.file_name)

    if not file_path:
        return

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    index_path = os.path.join(config['index_path'], f"{base_name}_faiss_index")

    try:
        rag_manager = RAGManager(file_path=file_path, index_path=index_path)
    except (FileNotFoundError, ValueError):
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    rag_manager.setup()
    
    interactive_manager = InteractiveManager(rag_manager)
    interactive_manager.run()

if __name__ == "__main__":
    main()