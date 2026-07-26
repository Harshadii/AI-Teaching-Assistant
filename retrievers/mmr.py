from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


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
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Vector Store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)


# Similarity Retriever
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("\n===== Similarity Results =====\n")

similarity_docs = similarity_retriever.invoke(
    "What is gradient descent?"
)

for doc in similarity_docs:
    print(doc.page_content)
    print("--------------------------------")


# MMR Retriever
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 5
    }
)

print("\n===== MMR Results =====\n")

mmr_docs = mmr_retriever.invoke(
    "What is gradient descent?"
)

for doc in mmr_docs:
    print(doc.page_content)
    print("--------------------------------")