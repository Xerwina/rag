import os
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter


# Connexion PostgreSQL
CONNECTION_STRING = "postgresql+psycopg2://postgres:Admin123$@localhost:5432/documents"
COLLECTION_NAME = "documents"

# Initialisation
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Récupération des fichiers PDF
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith(".pdf")]

# Chargement et découpage
all_docs = []
for file_name in pdf_files:
    file_path = os.path.join(current_dir, file_name)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    split_docs = splitter.split_documents(documents)
    all_docs.extend(split_docs)

# Insertion dans pgvector
if all_docs:
    db = PGVector.from_documents(
        all_docs,
        embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )
    print(f"{len(all_docs)} chunks issus des PDF ont été insérés dans la base vectorielle PostgreSQL.")
else:
    print("Aucun fichier PDF trouvé dans le dossier courant.")
