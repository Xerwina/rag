import os
from langchain_community.vectorstores.pgvector import PGVector
from langchain_huggingface import HuggingFaceEmbeddings   # nouveau package
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA


# --- Connexion PostgreSQL ---
CONNECTION_STRING = "postgresql+psycopg2://postgres:Admin123$@localhost:5432/rag"
COLLECTION_NAME = "documents"

# --- Initialisation ---
# Embeddings rapides & efficaces
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Splitter (500 tokens, overlap 50)
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# LLM : Mistral 7B via Ollama
llm = Ollama(model="mistral")  # ton Ollama doit avoir "mistral" disponible


# --- Récupération des fichiers PDF ---
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith(".pdf")]

all_docs = []
for file_name in pdf_files:
    file_path = os.path.join(current_dir, file_name)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    split_docs = splitter.split_documents(documents)
    all_docs.extend(split_docs)


# --- Insertion dans pgvector ---
if all_docs:
    db = PGVector.from_documents(
        all_docs,
        embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )
    print(f"{len(all_docs)} chunks issus des PDF ont été insérés dans la base vectorielle PostgreSQL.")
else:
    # Si la base est déjà remplie, on peut juste se reconnecter
    db = PGVector(
        embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
    )
    print("Aucun fichier PDF trouvé, utilisation de la base existante.")


# --- Création du pipeline RAG ---
retriever = db.as_retriever(search_kwargs={"k": 3})  # prend les 3 passages les plus proches
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# --- Exemple de requête ---
while True:
    query = input("\nPose une question (ou 'exit' pour quitter) : ")
    if query.lower() == "exit":
        break
    result = qa.run(query)
    print("\nRéponse :", result)
