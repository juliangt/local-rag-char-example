import os
import argparse
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

class RAGManager:
    def __init__(self, file_path, index_path, embedding_model="embeddinggemma", chat_model="gemma3:270m"):
        self.file_path = file_path
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.vector_store = None
        self.chain = None

        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"The file '{self.file_path}' does not exist.")

            # Validate file format if necessary (e.g., check extension)
            supported_extensions = ['.txt']
            if not any(self.file_path.endswith(ext) for ext in supported_extensions):
                raise ValueError(f"Unsupported file format. Please use one of {supported_extensions}.")

            # Placeholder for model validation
            # In a real-world scenario, you would check if the model files exist
            # or if the model names are valid by querying an API or a local registry.
            print(f"Attempting to initialize with embedding model: {self.embedding_model}")
            print(f"Attempting to initialize with chat model: {self.chat_model}")

        except (FileNotFoundError, ValueError) as e:
            self._log_error(e)
            print(f"Error: {e}")
            raise
        except Exception as e:
            self._log_error(e)
            print(f"An unexpected error occurred during initialization: {e}")
            raise

    def _log_error(self, error):
        with open("error.log", "a") as f:
            f.write(f"\n--- {type(error).__name__} ---\n")
            f.write(str(error))

    def _create_index(self):
        print(f"Creating index from {self.file_path} using {self.embedding_model}...")
        loader = TextLoader(self.file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vector_store = FAISS.from_documents(docs, embeddings)
        self.vector_store.save_local(self.index_path)
        print(f"Index saved to {self.index_path}")

    def _load_index(self):
        print(f"Loading index from {self.index_path} using {self.embedding_model}...")
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vector_store = FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)
        print("Index loaded.")

    def setup(self):
        if not os.path.exists(self.index_path) or not os.listdir(self.index_path):
            self._create_index()
        else:
            self._load_index()

        retriever = self.vector_store.as_retriever()
        llm = ChatOllama(model=self.chat_model)

        # Prompt to reformulate the question based on history
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Prompt for the final answer using the retrieved context
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        self.chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def ask(self, question, chat_history):
        if not self.chain:
            raise Exception("Chain not set up. Please run setup() first.")
        
        # Format history for the chain
        formatted_chat_history = []
        for human, ai in chat_history:
            formatted_chat_history.append(HumanMessage(content=human))
            formatted_chat_history.append(AIMessage(content=ai))

        result = self.chain.invoke({"input": question, "chat_history": formatted_chat_history})
        return result.get('answer', "I couldn't find an answer.")

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
        return  # Exit if initialization fails
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    rag_manager.setup()

    print(f"\nChat with {os.path.basename(file_path)}! Type 'exit' to quit.")
    chat_history = []
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() == 'exit':
                break
            if query.strip() == "":
                continue
            
            answer = rag_manager.ask(query, chat_history)
            print(f"\nAI: {answer}")
            chat_history.append((query, answer))

        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break



if __name__ == "__main__":
    main()