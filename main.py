from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory= "chroma_db",
    embedding_function=embedding_model
)

retrievers = vectorstore.as_retriever(
    search_type = "mmr" , 
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult" :  0.5
    }
)

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="conversational",
    max_new_tokens=512,
    temperature=0.7
)

llm = ChatHuggingFace(llm=llm)

#prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system create")

print("press 0 to exit")

while True:
    query = input("You : ")
    if query == "0":
        break

    docs = retrievers.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context":context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")