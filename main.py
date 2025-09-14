import os
import argparse
from rag_manager import RAGManager
from interactive_manager import InteractiveManager

def main():
    os.makedirs("docs", exist_ok=True)
    os.makedirs("indexes", exist_ok=True)

    parser = argparse.ArgumentParser(description="Chat with a document using RAG.")
    parser.add_argument("file_name", type=str, nargs='?', default=None, help="Name or path of the text file.")
    parser.add_argument("--index_path", type=str, help="Path to the FAISS index directory. Overrides default behavior.")
    parser.add_argument("--embedding_model", type=str, default="embeddinggemma", help="Name of the Ollama model to use for embeddings.")
    parser.add_argument("--chat_model", type=str, default="gemma3:270m", help="Name of the Ollama model to use for chat.")
    args = parser.parse_args()

    file_name = args.file_name
    if not file_name:
        files = [f for f in os.listdir("docs") if f.endswith('.txt')]
        if not files:
            print("No text files found in the 'docs' directory.")
            return
        elif len(files) == 1:
            file_name = files[0]
        else:
            print("Please choose a file to chat with:")
            for i, f in enumerate(files):
                print(f"[{i+1}] {f}")
            while True:
                try:
                    choice = int(input("Enter the number of the file: "))
                    if 1 <= choice <= len(files):
                        file_name = files[choice-1]
                        break
                    else:
                        print("Invalid number.")
                except ValueError:
                    print("Please enter a number.")

    file_path = file_name
    if not os.path.exists(file_path):
        file_path = os.path.join("docs", file_name)
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_name}' was not found in the current directory or in the 'docs' directory.")
            return

    index_path = args.index_path
    if not index_path:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        index_path = os.path.join("indexes", f"{base_name}_faiss_index")

    try:
        rag_manager = RAGManager(
            file_path=file_path, 
            index_path=index_path, 
            embedding_model=args.embedding_model,
            chat_model=args.chat_model
        )
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