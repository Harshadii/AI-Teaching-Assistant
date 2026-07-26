import sys
if sys.platform != "win32":
    try:
        __import__('pysqlite3')
        import sys
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    except ImportError:
        pass

import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Load env variables from .env if present
load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Teaching Assistant", page_icon="🤖", layout="wide")

# Custom CSS for rich, professional resume styling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* ===========================
   Hero Title
=========================== */

.title-container{
    position:relative;
    overflow:hidden;
    padding:2.8rem 2rem;
    border-radius:22px;

    background:linear-gradient(
        135deg,
        #5B5FEF 0%,
        #7C4DFF 35%,
        #00C6FF 100%
    );

    background-size:300% 300%;
    animation:gradientMove 10s ease infinite;

    text-align:center;
    color:#fff;

    margin-bottom:2rem;

    box-shadow:
    0 15px 40px rgba(92,95,239,.25);

    border:1px solid rgba(255,255,255,.15);
}

/* Floating Glow */

.title-container::before{

content:"";

position:absolute;

top:-60%;

left:-40%;

width:180%;

height:180%;

background:

radial-gradient(circle,
rgba(255,255,255,.18) 0%,
transparent 65%);

animation:rotateGlow 12s linear infinite;

pointer-events:none;

}

.title-container h1{

font-size:3rem !important;

font-weight:800 !important;

margin-bottom:.5rem !important;

letter-spacing:-1px;

color:white;

animation:floatTitle 4s ease-in-out infinite;

}

.title-container p{

font-size:1.08rem;

opacity:.95;

font-weight:400;

letter-spacing:.4px;

}


/* ===========================
   Metric Cards
=========================== */

.metric-grid{

display:grid;

grid-template-columns:repeat(4,1fr);

gap:18px;

margin-bottom:2rem;

}

.metric-card{

background:rgba(255,255,255,.06);

backdrop-filter:blur(18px);

-webkit-backdrop-filter:blur(18px);

border:1px solid rgba(255,255,255,.08);

border-radius:18px;

padding:1.4rem;

text-align:center;

transition:.35s ease;

position:relative;

overflow:hidden;

}

.metric-card::before{

content:"";

position:absolute;

top:0;

left:0;

width:100%;

height:4px;

background:linear-gradient(
90deg,
#6C63FF,
#00C6FF
);

}

.metric-card:hover{

transform:

translateY(-8px)

scale(1.03);

border-color:#6C63FF;

box-shadow:

0 18px 40px rgba(92,95,239,.22);

}

.metric-label{

font-size:.78rem;

text-transform:uppercase;

letter-spacing:1.4px;

color:#A1A7C4;

margin-bottom:.5rem;

}

.metric-value{

font-size:1.8rem;

font-weight:700;

background:linear-gradient(
90deg,
#6C63FF,
#00C6FF
);

-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

}


/* ===========================
   Sidebar
=========================== */

.sidebar-section-header{

font-size:1rem;

font-weight:700;

color:white;

border-radius:12px;

background:linear-gradient(
90deg,
rgba(108,99,255,.18),
rgba(0,198,255,.15)
);

border-left:4px solid #6C63FF;

margin-top:1rem;

margin-bottom:.7rem;

}


/* ===========================
   Status Badge
=========================== */

.status-badge{

display:inline-flex;

align-items:center;

gap:6px;

padding:.35rem .8rem;

border-radius:30px;

font-size:.8rem;

font-weight:600;

}

.status-ready{

background:rgba(34,197,94,.15);

color:#22C55E;

border:1px solid rgba(34,197,94,.3);

}

.status-empty{

background:rgba(239,68,68,.15);

color:#EF4444;

border:1px solid rgba(239,68,68,.3);

}


/* ===========================
   Animations
=========================== */

@keyframes gradientMove{

0%{
background-position:0% 50%;
}

50%{
background-position:100% 50%;
}

100%{
background-position:0% 50%;
}

}

@keyframes floatTitle{

0%{
transform:translateY(0);
}

50%{
transform:translateY(-6px);
}

100%{
transform:translateY(0);
}

}

@keyframes rotateGlow{

0%{
transform:rotate(0deg);
}

100%{
transform:rotate(360deg);
}
}
</style>
""", unsafe_allow_html=True)

DB_DIR = "chroma_db"

# -------------------------------------------------------------
# API Key Setup & Validation
# -------------------------------------------------------------
# First priority: env var, Second priority: sidebar user input
env_api_key = os.environ.get("MISTRAL_API_KEY", "")

# -------------------------------------------------------------
# Sidebar Setup
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 1.5rem;'><h2 style='margin:0;'>⚙️Settings</h2></div>", unsafe_allow_html=True)
    
    # API Key Input
    st.markdown("<div class='sidebar-section-header'>🔐 Authentication</div>", unsafe_allow_html=True)
    if env_api_key:
        st.success("🤖 AI Engine Connected")
        api_key = env_api_key
    else:
        api_key = st.text_input("🔐 Connect Mistral AI", type="password", help="Mistral API Key is required for embeddings and LLM generation.")
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key
            st.success("API Key loaded successfully!")
            st.rerun()

    # Document Upload Section
    st.markdown("<div class='sidebar-section-header'>📥 Import PDFs </div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 Choose Study Document", type="pdf")
    chunk_size = 1000
    chunk_overlap = 200
    retriever_type = "Maximal Marginal Relevance (MMR)"
    k_value = 5
    lambda_value = 0.5

    # Trigger processing
    process_btn = st.button("⚡ Generate Vector Embeddings", type="primary", use_container_width=True)

    # Maintenance Buttons
    st.markdown("<div class='sidebar-section-header'>🔄 Reset Settings</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat cleared!")
            st.rerun()
    with c2:
        if st.button("♻️ Start Fresh", use_container_width=True):
            if st.session_state.vectorstore is not None:
                try:
                    st.session_state.vectorstore.delete_collection()
                except Exception:
                    pass
            st.session_state.vectorstore = None
            
            import gc
            gc.collect()
            
            if os.path.exists(DB_DIR):
                try:
                    shutil.rmtree(DB_DIR)
                except Exception:
                    # On Windows, ignore file lock error as the collection is already deleted
                    pass
                    
            st.session_state.book_name = None
            st.session_state.total_pages = 0
            st.session_state.total_chunks = 0
            st.session_state.messages = []
            st.success("DB Reset!")
            st.rerun()

# -------------------------------------------------------------
# Load Vector Store Helper
# -------------------------------------------------------------
def load_existing_db():
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        if os.environ.get("MISTRAL_API_KEY"):
            try:
                embeddings = MistralAIEmbeddings(model="mistral-embed")
                return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            except Exception as e:
                st.sidebar.error(f"Error loading VectorDB: {e}")
    return None

# -------------------------------------------------------------
# Initialize Session State
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_existing_db()
if "book_name" not in st.session_state:
    st.session_state.book_name = "📚 Knowledge Base" if st.session_state.vectorstore else None
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# -------------------------------------------------------------
# Build Vector Store from uploaded file
# -------------------------------------------------------------
if process_btn:
    if not os.environ.get("MISTRAL_API_KEY"):
        st.error("🔐 Please connect your Mistral AI API key to continue.")
    elif not uploaded_file:
        st.error("📂 Please select a PDF file before proceeding.")
    else:
        with st.spinner("Processing document... Parsing PDF, splitting text, and indexing embeddings..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Load and split
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                chunks = splitter.split_documents(docs)

                # Overwrite/Clear DB safely without causing Windows lock errors
                if st.session_state.vectorstore is not None:
                    try:
                        # Clear existing collection first
                        st.session_state.vectorstore.delete_collection()
                        # Add new documents to same store
                        st.session_state.vectorstore.add_documents(chunks)
                        vectorstore = st.session_state.vectorstore
                    except Exception:
                        # Fallback: force garbage collection and recreate
                        st.session_state.vectorstore = None
                        import gc
                        gc.collect()
                        
                        try:
                            if os.path.exists(DB_DIR):
                                shutil.rmtree(DB_DIR)
                        except Exception:
                            pass
                            
                        embeddings = MistralAIEmbeddings(model="mistral-embed")
                        vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=DB_DIR
                        )
                else:
                    try:
                        if os.path.exists(DB_DIR):
                            shutil.rmtree(DB_DIR)
                    except Exception:
                        pass
                        
                    embeddings = MistralAIEmbeddings(model="mistral-embed")
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=DB_DIR
                    )

                # Update session state
                st.session_state.vectorstore = vectorstore
                st.session_state.book_name = uploaded_file.name
                st.session_state.total_pages = len(docs)
                st.session_state.total_chunks = len(chunks)
                st.session_state.messages = [] # Clear history on new document

                # Clean up tmp file
                os.remove(tmp_file_path)
                st.sidebar.success("🧠 AI Knowledge Base Created Successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ An error occurred during document processing: {e}")

# -------------------------------------------------------------
# Main Page View
# -------------------------------------------------------------
# Beautiful Header Card
st.markdown("""
<div class='title-container'>
    <h1>🧠 AI Teaching Assistant</h1>
    <p>Upload your document and chat with your files </p>
</div>
""", unsafe_allow_html=True)

# KPI Metric Cards
st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
db_status_badge = "<span class='status-badge status-ready'>Ready ✅</span>" if st.session_state.vectorstore else "<span class='status-badge status-empty'>Empty ❌</span>"
active_book = st.session_state.book_name if st.session_state.book_name else "No Document Loaded"
pages_count = str(st.session_state.total_pages) if st.session_state.total_pages > 0 else " 0 "
chunks_count = str(st.session_state.total_chunks) if st.session_state.total_chunks > 0 else " 0 "

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Database Status</div>
        <div style='margin-top:0.4rem;'>{db_status_badge}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>Doc Name</div>
        <div class='metric-value' style='font-size:1.1rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-top:0.3rem;'>{active_book}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>📑 Pages</div>
        <div class='metric-value'>{pages_count}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>📦 Segments</div>
        <div class='metric-value'>{chunks_count}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# Conversation Section
# -------------------------------------------------------------
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If assistant has sources, render them
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Source References"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {idx} | Page {src['page']}** (Relevance Match)")
                    st.code(src["content"], language="text")

# Chat input
query = st.chat_input("Ask a question about the document...")

if query:
    # 1. API key validation
    if not os.environ.get("MISTRAL_API_KEY"):
        st.error("❌ Please enter your Mistral API Key in the sidebar to ask questions.")
        st.stop()
        
    # 2. Database validation
    if not st.session_state.vectorstore:
        st.warning("⚠️ RAG database is empty. Please upload a PDF and click 'Build RAG Vector Store' in the sidebar.")
        st.stop()

    # Append and show User Message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Show Assistant Response with Spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching document & generating answer..."):
            try:
                # 3. Conversational Query Reformulation (standalone question)
                query_to_retrieve = query
                if len(st.session_state.messages) > 1:
                    try:
                        # Build langchain conversation history list
                        lc_history = []
                        for m in st.session_state.messages[:-1]: # exclude current user message
                            if m["role"] == "user":
                                lc_history.append(HumanMessage(content=m["content"]))
                            else:
                                lc_history.append(AIMessage(content=m["content"]))
                        
                        reformulate_prompt = ChatPromptTemplate.from_messages([
                            ("system", "Given a conversation history and a follow-up question, reformulate the follow-up question to be a standalone question that has all context needed for document retrieval. Return ONLY the standalone question text, do not add any conversational text or formatting."),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{question}")
                        ])
                        
                        llm_for_query = ChatMistralAI(model="mistral-small-latest", temperature=0)
                        standalone_query_response = llm_for_query.invoke(
                            reformulate_prompt.format_messages(
                                chat_history=lc_history,
                                question=query
                            )
                        )
                        standalone_query = standalone_query_response.content.strip()
                        if standalone_query:
                            query_to_retrieve = standalone_query
                    except Exception as e:
                        # Fail-safe: fall back to original query
                        pass

                # 4. Retrieval execution
                if retriever_type == "Maximal Marginal Relevance (MMR)":
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": k_value,
                            "fetch_k": max(20, k_value * 2),
                            "lambda_mult": lambda_value
                        }
                    )
                else:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={
                            "k": k_value
                        }
                    )

                retrieved_docs = retriever.invoke(query_to_retrieve)
                
                # Format context
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Format sources for UI
                sources_for_ui = []
                for doc in retrieved_docs:
                    page = doc.metadata.get("page", 0) + 1 # pypdf pages are 0-indexed
                    sources_for_ui.append({
                        "page": page,
                        "content": doc.page_content
                    })

                # 5. Core Chat Generation Prompt
                prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """You are an expert AI document assistant. 
Use the provided Context chunks to answer the user's question. 

Be thorough, precise, and professional. Ensure your answer is well-structured and easy to read (use lists/bullet points when appropriate).

If the answer is NOT explicitly supported by the context, state: "I could not find the answer in the document." Do NOT attempt to answer using external information.

---
CONTEXT:
{context}"""
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])

                # Build full LangChain message history
                lc_history = []
                for m in st.session_state.messages[:-1]: # exclude current user message
                    if m["role"] == "user":
                        lc_history.append(HumanMessage(content=m["content"]))
                    else:
                        lc_history.append(AIMessage(content=m["content"]))

                final_messages = prompt.invoke({
                    "context": context,
                    "chat_history": lc_history,
                    "question": query
                })

                llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)
                response = llm.invoke(final_messages)
                answer_content = response.content

                # Render final answer
                st.markdown(answer_content)
                
                # Render sources
                with st.expander("📄 Source References"):
                    for idx, src in enumerate(sources_for_ui, 1):
                        st.markdown(f"**Chunk {idx} | Page {src['page']}** (Relevance Match)")
                        st.code(src["content"], language="text")

                # Store response in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_content,
                    "sources": sources_for_ui
                })
                
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
