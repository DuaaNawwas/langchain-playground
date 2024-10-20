from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

file_path = "./data/test.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

import getpass
import os

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb

persistent_client = chromadb.HttpClient(host="http://localhost:8000")
collection = persistent_client.get_or_create_collection("py-coll")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
from uuid import uuid4

uuids = [str(uuid4()) for _ in range(len(splits))]

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    client=persistent_client,
    collection_name="py-coll",
    embedding_function=embedding
)

vectorstore.add_documents(documents=splits, ids=uuids)

retriever = vectorstore.as_retriever()


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "عاي قسم لازم ادخل من الموقع"})

print(results)