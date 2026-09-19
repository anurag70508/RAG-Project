from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyMuPDFLoader("document loaders/GRU.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

print(len(chunks))
print()
print(chunks[0].page_content)
