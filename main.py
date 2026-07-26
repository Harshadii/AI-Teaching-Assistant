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
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# Embedding
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)


# Vector Store
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)


# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# LLM
llm = ChatMistralAI(
    model="mistral-small-latest"
)


# Prompt
prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say:
"I could not find the answer in the document."
"""
        ),

        (
            "human",
            """
Context:

{context}

Question:

{question}
"""
        )

    ]
)


print("RAG system created")

print("Press 0 to exit")


while True:

    query = input("You : ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(final_prompt)

    print(f"\nAI : {response.content}\n")