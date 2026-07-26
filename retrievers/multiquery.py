from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import ChatMistralAI


# Documents
docs = [

    Document(
        page_content="Gradient descent is an optimization algorithm used in machine learning."
    ),

    Document(
        page_content="Gradient descent minimizes the loss function."
    ),

    Document(
        page_content="Gradient descent is an optimization that minimizes the loss by updating weights."
    ),

    Document(
        page_content="Neural networks use gradient descent for training."
    ),

    Document(
        page_content="Support Vector Machines are supervised learning algorithms."
    )

]


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# Vector Store
vectorstore = Chroma.from_documents(
    docs,
    embeddings
)


# Retriever
retriever = vectorstore.as_retriever()


# LLM
llm = ChatMistralAI(
    model="mistral-small-latest"
)


# Multi Query Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)


# Query
query = "What is gradient descent?"


# Retrieve
docs = multi_query_retriever.invoke(query)


print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)
    print("--------------------------------")