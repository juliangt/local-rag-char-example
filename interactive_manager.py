import os
import shutil
from config import config

class InteractiveManager:
    def __init__(self, rag_manager):
        self.rag_manager = rag_manager
        self.chat_history = []

    def run(self):
        if not self.rag_manager.active_document:
            print("\nNo documents found in 'docs' folder. Please add documents and restart.")
            return
        print(f"\nChat with {self.rag_manager.active_document}! Type '/help' for a list of commands.")
        while True:
            try:
                query = input("\nYou: ").strip()
                if not query:
                    continue

                if query.startswith('/'):
                    self.handle_command(query)
                else:
                    answer = self.rag_manager.ask(query, self.chat_history)
                    print(f"\nAI: {answer}")
                    if "The index is not set up" not in answer and config.get('replay_history', True):
                        self.chat_history.append((query, answer))
                        if len(self.chat_history) > config.get('max_replay_history', 5):
                            self.chat_history.pop(0)

            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

    def handle_command(self, query):
        command, *args = query.lower().split()
        if command == '/exit':
            raise SystemExit
        elif command == '/clear':
            self.clear_index()
        elif command == '/reindex':
            self.reindex()
        elif command == '/list_docs':
            self.list_docs()
        elif command == '/use_doc':
            self.use_doc(args)
        elif command == '/help':
            self.show_help()
        else:
            self.show_suggestions(command)

    def clear_index(self):
        index_path = self.rag_manager.index_path
        if os.path.exists(index_path):
            try:
                shutil.rmtree(index_path)
                os.makedirs(index_path, exist_ok=True)
                self.rag_manager.vector_store = None
                self.rag_manager.chain = None
                print(f"Index at {index_path} has been cleared. Please run `/reindex` to create a new index.")
            except OSError as e:
                print(f"Error clearing index: {e}")
        else:
            print("No index found to clear.")

    def reindex(self):
        print("Re-indexing...")
        self.rag_manager._create_index()
        self.rag_manager.setup()
        print("Re-indexing complete.")

    def list_docs(self):
        docs = self.rag_manager.list_documents()
        if not docs:
            print("No documents found.")
            return

        print("Available documents:")
        for doc in docs:
            if doc == self.rag_manager.active_document:
                print(f"  - {doc} (active)")
            else:
                print(f"  - {doc}")

    def use_doc(self, args):
        if len(args) != 1:
            print("Usage: /use_doc <document_name>")
            return

        doc_name = args[0]
        self.rag_manager.switch_document(doc_name)

    def switch_model(self, args):
        if len(args) != 2:
            print("Usage: /model <embedding_model|chat_model> <model_name>")
            return

        model_type, model_name = args
        if model_type == "embedding_model":
            self.rag_manager.embedding_model = model_name
            print(f"Embedding model switched to: {model_name}")
        elif model_type == "chat_model":
            self.rag_manager.chat_model = model_name
            print(f"Chat model switched to: {model_name}")
        else:
            print("Invalid model type. Use 'embedding_model' or 'chat_model'.")
            return
        
        self.reindex()

    def show_help(self):
        print("""
Available commands:
  /clear          - Clear the index for the active document.
  /reindex        - Re-create the index for the active document.
  /list_docs      - List available documents.
  /use_doc <name>   - Switch to a different document.
  /model <type> <name> - Switch the embedding or chat model.
                    <type>: embedding_model | chat_model
                    <name>: name of the model
  /help           - Show this help message.
  /exit           - Exit the chat.
""")

    def show_suggestions(self, command):
        suggestions = {
            "/clear": "Did you mean /clear?",
            "/reindex": "Did you mean /reindex?",
            "/list_docs": "Did you mean /list_docs?",
            "/use_doc": "Did you mean /use_doc <document_name>?",
            "/help": "Did you mean /help?",
            "/exit": "Did you mean /exit?",
        }
        suggestion = suggestions.get(command, f"Unrecognized command: {command}. Try '/help'.")
        print(suggestion)
