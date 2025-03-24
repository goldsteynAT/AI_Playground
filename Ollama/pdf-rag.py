# 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = "./data/BOI.pdf"
model = "phi4:latest"

# == Local PDF file uploads ==
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("Done loading...")

else:
    print("Upload a PDF file")

# Testing if it works (outputs the first 100 characters of the first page)
# content = data[0].page_content
# print(content[:100])

# Extract Text from PDF Files and split into small chunks
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# == Split and chunk ==
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("Done splitting...")

# Testing if it works (outputs the number of chunks and an example chunk)
# print(f"Number of chunks: {len(chunks)}")
# print(f"Example chunk: {chunks[0]}")

# == Add to vector database ==
# First we need to create embeddings - we use nomic
import ollama
ollama.pull("nomic-embed-text")
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text")
    collection_name="simple-rag"
)
print("Done adding to vector database...")

# == Retrieval ==
# Used to create structured prompts for chat-based and general LLM interactions.
from langchain.prompts import ChatPromptTemplate, PromptTemplate
# Converts LLM responses into plain strings, useful for extracting clean text.
from langchain_core.output_parsers import StrOutputParser
# Interface to interact with Ollama's chat models.
from langchain_ollama import ChatOllama
# Passes data through a chain without modification, useful for debugging or simple pipelines.
from langchain_core.runnables import RunnablePassthrough
# Generates multiple queries to enhance retrieval from vector databases.
from langchain.retrievers.multi_query import MultiQueryRetriever

# set up our model to use
llm = ChatOllama(model=model)


QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

template = """Answer the questino based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

--> 2:20:08