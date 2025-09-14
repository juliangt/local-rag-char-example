import os
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

            supported_extensions = ['.txt']
            if not any(self.file_path.endswith(ext) for ext in supported_extensions):
                raise ValueError(f"Unsupported file format. Please use one of {supported_extensions}.")

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
