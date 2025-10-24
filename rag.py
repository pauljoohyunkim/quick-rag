from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

class RAG:
    def __init__(self, pdf_path, model="llama3.2:3b", chunk_size=1200, chunk_overlap=300):
        self.pdf_path = pdf_path
        self.model = model

        loader = PyPDFLoader(file_path=pdf_path)
        self.data = loader.load()

        del loader

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = text_splitter.split_documents(self.data)
        print(f"Number of chunks: {len(self.chunks)}")
        print(f"First chunk: {self.chunks[0]}")
    
    def embed_data(self, embedding_model="nomic-embed-text", collection_name="rag-system", base_url="http://localhost:11434"):
        self.vector_db = Chroma.from_documents(
            documents=self.chunks,
            embedding=OllamaEmbeddings(model=embedding_model, base_url=base_url),
            collection_name=collection_name
        )
        print("Added to vector database.")

        self.llm = ChatOllama(model=self.model, base_url=base_url)
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five different versions
            of the given user question to retrieve relevant documents from a vector database.
            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the similarity search.
            Provide these alternative separated by new lines
            Original question: {question}"""
        )
        self.retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(), self.llm, prompt=QUERY_PROMPT
        )
        template = """Answer the question ONLY on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question):
        return self.chain.invoke(input=(question,))
