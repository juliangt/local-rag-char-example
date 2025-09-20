import os
import argparse
from rag_manager import RAGManager
from interactive_manager import InteractiveManager
from config import config
from file_manager import FileManager

def main():
    os.makedirs(config['index_path'], exist_ok=True)

    file_manager = FileManager()
    file_paths = file_manager.get_all_file_paths()

    if not file_paths:
        print("No documents found to process. Please add supported files to the 'docs' directory.")
        return

    # Use a generic index name for the combined index
    index_name = "multi_doc_index"
    index_path = os.path.join(config['index_path'], f"{index_name}_faiss_index")

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