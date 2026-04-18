from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

_vector_store = None


def build_vector_store(kb_path: str) -> None:
    global _vector_store
    loader = DirectoryLoader(kb_path, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _vector_store = Chroma.from_documents(chunks, embeddings)


def retrieve_chunks(query: str, k: int = 4) -> list:
    if _vector_store is None:
        raise RuntimeError("Call build_vector_store() before retrieve_chunks()")
    results = _vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
