from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

data = PyMuPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI that summarizes the text in brief."),
     ("human", "{data}")]
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# prompt = template.format_messages(data = docs[0].page_content)
prompt = template.format_messages(data = docs)

result = model.invoke(prompt)

print(result.text)