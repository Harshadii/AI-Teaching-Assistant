import sys
if sys.platform != "win32":
    try:
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass
else:
    # On Windows, enforce UTF-8 for console output to avoid encoding errors with ligatures/special characters
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma


# -----------------------------
# Load PDF
# -----------------------------

loader = PyPDFLoader("document loaders/deeplearning.pdf")

docs = loader.load()


# -----------------------------
# Split into Chunks
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)


# -----------------------------
# Embedding Model (Mistral)
# -----------------------------

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)


# -----------------------------
# Store into ChromaDB
# -----------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Database Created Successfully!")


# ====================================================
# Similarity Search
# ====================================================

print("\n========== Similarity Search ==========\n")

results = vectorstore.similarity_search(
    "What is Deep Learning?",
    k=2
)

for r in results:
    print(r.page_content)
    print(r.metadata)
    print("--------------------------------------")


# ====================================================
# Retriever
# ====================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k":2}
)

docs = retriever.invoke(
    "Explain Neural Networks"
)

print("\n========== Retriever ==========\n")

for d in docs:
    print(d.page_content)
    print(d.metadata)
    print("--------------------------------------")