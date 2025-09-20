import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from config import config

class RAGManager:
    def __init__(self, file_paths, index_path):
        self.file_paths = file_paths
        self.index_path = index_path
        self.embedding_model = config['embedding_model_path']
        self.chat_model = config['llm_model_path']
        self.vector_store = None
        self.chain = None

        try:
            if not self.file_paths:
                raise FileNotFoundError("No files provided to process.")

            for file_path in self.file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"The file '{file_path}' does not exist.")

                supported_extensions = config['supported_extensions']
                if not any(file_path.endswith(ext) for ext in supported_extensions):
                    raise ValueError(f"Unsupported file format for {os.path.basename(file_path)}. Please use one of {supported_extensions}.")

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
        print(f"Creating index from {len(self.file_paths)} file(s) using {self.embedding_model}...")
        
        loader_map = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".md": UnstructuredMarkdownLoader,
            ".docx": Docx2txtLoader,
        }
        
        all_documents = []
        for file_path in self.file_paths:
            file_extension = os.path.splitext(file_path)[1]
            loader_class = loader_map.get(file_extension)

            if not loader_class:
                print(f"Warning: No loader found for file extension {file_extension}. Skipping {os.path.basename(file_path)}.")
                continue

            try:
                loader = loader_class(file_path)
                documents = loader.load()
                all_documents.extend(documents)
                print(f"Loaded {os.path.basename(file_path)}.")
            except Exception as e:
                print(f"Error loading {os.path.basename(file_path)}: {e}")

        if not all_documents:
            print("No documents were loaded. Index not created.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap'])
        docs = text_splitter.split_documents(all_documents)
        
        embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vector_store = FAISS.from_documents(docs, embeddings)
        self.vector_store.save_local(self.index_path)
        print(f"Combined index for {len(self.file_paths)} file(s) saved to {self.index_path}")


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

        retriever = self.vector_store.as_retriever(k=config['k_retriever'])
        llm = ChatOllama(model=self.chat_model, temperature=config['temperature'], max_new_tokens=config['max_new_tokens'], n_ctx=config['n_ctx'], n_gpu_layers=config['n_gpu_layers'], verbose=config['verbose'])

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),

            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        self.chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def ask(self, question, chat_history):
        if not self.chain:
            return "The index is not set up. Please run the `/reindex` command."

        formatted_chat_history = []
        for human, ai in chat_history:
            formatted_chat_history.append(HumanMessage(content=human))
            formatted_chat_history.append(AIMessage(content=ai))

        result = self.chain.invoke({"input": question, "chat_history": formatted_chat_history})
        return result.get('answer', "I couldn't find an answer.")
