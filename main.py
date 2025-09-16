import os
from rag_manager import RAGManager
from interactive_manager import InteractiveManager
from config import config

def main():
    # Ensure the root index directory exists
    index_path = config.get('index_path')
    if index_path:
        os.makedirs(index_path, exist_ok=True)

    try:
        rag_manager = RAGManager()
        if not rag_manager.active_document:
            return
    except (FileNotFoundError, ValueError) as e:
        print(f"Failed to initialize RAG Manager: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        return

    rag_manager.setup()
    
    interactive_manager = InteractiveManager(rag_manager)
    interactive_manager.run()

if __name__ == "__main__":
    main()
