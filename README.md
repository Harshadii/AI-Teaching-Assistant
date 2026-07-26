
## 🤖 AI Teaching Assistant (RAG)

An intelligent AI-powered Teaching Assistant built using Retrieval-Augmented Generation (RAG) that allows students to upload study materials (PDFs) and ask natural language questions. Instead of relying only on the language model's knowledge, the assistant retrieves relevant information from uploaded documents to generate accurate, context-aware answers.
## ✨ Features

📄 Upload one or multiple PDF documents

💬 Ask questions in natural language

🧠 Retrieval-Augmented Generation (RAG)

🔍 Semantic search using vector embeddings

⚡ Fast document retrieval with FAISS/ChromaDB

🤖 Mistral LLM integration

🎨 Modern Streamlit interface

📚 Context-aware and reliable responses
## 🏗️ Architecture


                            Workflow

                        PDF Documents
                            │
                            ▼
                        Document Loader
                            │
                            ▼
                        Text Splitter
                            │
                            ▼
                        Embedding Model
                            │
                            ▼
                Vector Database (FAISS/ChromaDB)
                                ▲
                                │
                    User Question → Retriever
                                │
                                ▼
                        Mistral LLM
                                │
                                ▼
                    AI Generated Answer
## 🛠️ Tech Stack

    Category               Technologies
    Language               Python
    Frontend               Streamlit
    Framework              LangChain
    LLM                    Google Gemini
    Embeddings             Hugging Face Embeddings
    Vector Database        ChromaDB
    Document Processing    PyPDF-Text Splitter
    Environment            python-dotenv## 📂 Project Structure

```text
AI-Teaching-Assistant
│
├── 📁 .devcontainer/             
├── 📁 chroma_db/                  
├── 📁 document_loaders/            
├── 📁 retrievers/                 
├── 📁 vector_store/               
├── 📁 venv/                        
├── 📄 .env                        
├── 📄 .gitignore                  
├── 📄 app.py                      
├── 📄 create_database.py           
├── 📄 main.py                     
├── 📄 requirements.txt            
├── 📄 README.md                   
└── 📄 LICENSE                     
```
## 🚀 Installation



#### Clone the repository
```bash
Copy from my profile 
```
#### Navigate to the project
```bash
cd AI-Teaching-Assistant
```
#### Create a virtual environment
```bash
python -m venv venv
```
#### Activate it
```bash
Windows

venv\Scripts\activate
```
```bash
Linux / macOS

source venv/bin/activate
```
#### Install Dependencies 
```bash
pip install -r requirements.txt
```
#### Create a .env File
```bash
MISTRAL_API_KEY=your_api_key_here
```
#### Run The Application
```bash
streamlit run app.py

```





## 💡 How It Works
    1. Upload PDF study materials.
    2. Documents are split into smaller chunks.
    3. Chunks are converted into embeddings.
    4. Embeddings are stored in a vector database.
    5. The retriever finds the most relevant content.
    6. Mistral generates an answer using the retrieved context.
    7. The assistant returns an accurate response.
## Screenshots

  ![App Screenshot](https://github.com/Harshadii/AI-Teaching-Assistant/blob/main/document%20loaders/WorkFlow.png)

  ![App Screenshot](https://github.com/Harshadii/AI-Teaching-Assistant/blob/main/document%20loaders/Demo%201.png)

  ![App Screenshot](https://github.com/Harshadii/AI-Teaching-Assistant/blob/main/document%20loaders/Demo2.png)


## 🎯 Future Improvements
    Voice interaction
    Image understanding
    Multi-language support
    Chat history
    User authentication
    Cloud deployment
    Citation highlighting
## 🤝 Contributing
    Contributions, suggestions, and feature requests are welcome.

## 👨‍💻 Author
    Harsh Aditya

    B.Tech CSE (Data Science)

    Server live : https://ai-teaching-assistant-harshadii.streamlit.app/

    If you found this project helpful, consider giving it a ⭐ on GitHub.
