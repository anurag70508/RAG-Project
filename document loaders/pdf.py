from langchain_community.document_loaders import PyMuPDFLoader

data = PyMuPDFLoader("document loaders/GRU.pdf")

docs = data.load()

# print(len(docs))  -> 15

print(docs[14])
