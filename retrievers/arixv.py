from langchain_community.retrievers import ArxivRetriever
# Create Retriever

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True
)
# Query Arxiv


docs = retriever.invoke(
    "large language models"
)
# Print Results

for i, doc in enumerate(docs):

    print(f"\n========== Result {i+1} ==========\n")

    print("Title   :", doc.metadata.get("Title"))
    print("Authors :", doc.metadata.get("Authors"))
    print("Summary :")
    print(doc.page_content)

    print("\n-----------------------------------------")
