import os
import argparse
from rag_manager import RAGManager
from interactive_manager import InteractiveManager
from config import config

def main():
    os.makedirs(config['docs_path'], exist_ok=True)
    os.makedirs(config['index_path'], exist_ok=True)

    parser = argparse.ArgumentParser(description="Chat with a document using RAG.")
    parser.add_argument("file_name", type=str, nargs='?', default=None, help="Name or path of the text file.")
    args = parser.parse_args()

    file_name = args.file_name
    if not file_name:
        files = [f for f in os.listdir(config['docs_path']) if f.endswith('.txt')]
        if not files:
            print(f"No text files found in the '{config['docs_path']}' directory.")
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
        file_path = os.path.join(config['docs_path'], file_name)
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_name}' was not found in the current directory or in the '{config['docs_path']}' directory.")
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