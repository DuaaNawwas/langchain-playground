from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

PDF_PATH = "data"
def load_pdfs():
    loader = PyPDFDirectoryLoader(PDF_PATH)
    docs = loader.load()
    print(len(docs))

def split_docs(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def save_to_chroma(splits: list[Document]):
    vectorstore = Chroma(collection_name='py-coll')
    vectorstore.add_documents(splits)
    # vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), collection_name='py-coll')
    retriever = vectorstore.as_retriever()
    return retriever

def run():
   docs = load_pdfs()
   splits = split_docs(docs)
   print(len(splits))
   save_to_chroma(splits)
    



# def save_to_chroma(splits: list[Document]):