from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma


# -----------------------------
# Create Documents
# -----------------------------

docs = [

    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"}
    ),

    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"}
    ),

    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"}
    )

]


# -----------------------------
# Embedding Model
# -----------------------------

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)


# -----------------------------
# Create Vector Store
# -----------------------------

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

print("Vector DB Created Successfully!")


# =====================================================
# Similarity Search
# =====================================================

print("\n========== Similarity Search ==========\n")

result = vectorstore.similarity_search(
    "What is used for data analysis?",
    k=2
)

for r in result:
    print("Page Content : ", r.page_content)
    print("Metadata     : ", r.metadata)
    print("--------------------------------------")


# =====================================================
# Retriever
# =====================================================

print("\n========== Retriever ==========\n")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

docs = retriever.invoke(
    "Explain deep learning"
)

for d in docs:
    print("Page Content : ", d.page_content)
    print("Metadata     : ", d.metadata)
    print("--------------------------------------")
