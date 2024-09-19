import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

folder_path = "db"
pdf_folder = "pdf"

# Initialize the LLM model
cached_llm = Ollama(model="llama3")

# Initialize the embedding model
embedding = NomicEmbeddings(model="nomic-embed-text-v1.5")

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150, length_function=len, is_separator_regex=False
)

# Prompt template
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] Vous êtes un assistant technique expert dans la recherche de documents de l'Institut national de statistique. Répondez en français à toutes les questions posées.Ne recherchez pas d'informations sur Internet. [/INST] </s>
    [INST] {input}
           Contexte : {context}
           Réponse :
    [/INST]
"""
)


def initialize_vector_store():
    return Chroma(persist_directory=folder_path, embedding_function=embedding)


def create_chain():
    vector_store = initialize_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.5},
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    return create_retrieval_chain(retriever, document_chain)


@app.route("/direct_ai", methods=["POST"])
def direct_ai_post():
    json_content = request.json
    query = json_content.get("query")
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    response = cached_llm.invoke(query)
    return jsonify({"answer": response})


@app.route("/ask_pdf", methods=["POST"])
def ask_pdf_post():
    json_content = request.json
    query = json_content.get("query")
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    chain = create_chain()
    result = chain.invoke({"input": query})

    sources = [
        {"source": doc.metadata.get("source", "N/A"),
         "page_content": doc.page_content}
        for doc in result.get("context", [])
    ]

    return jsonify({"answer": result.get("answer", "No answer found"), "sources": sources})


@app.route("/pdf", methods=["POST"])
def pdf_post():
    file = request.files["file"]
    if not file:
        return jsonify({"error": "File parameter is missing"}), 400

    file_name = file.filename
    save_file = os.path.join(pdf_folder, file_name)
    file.save(save_file)

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()

    chunks = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    vector_store.persist()

    return jsonify({
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    })


@app.route("/index_pdf_folder", methods=["POST"])
def index_pdf_folder():
    if not os.path.exists(pdf_folder):
        return jsonify({"error": f"The folder '{pdf_folder}' does not exist."}), 400

    all_docs = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)
            loader = PDFPlumberLoader(file_path)
            docs = loader.load_and_split()
            all_docs.extend(docs)

    chunks = text_splitter.split_documents(all_docs)

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )
    vector_store.persist()

    return jsonify({
        "status": "Successfully Indexed All PDFs",
        "total_docs": len(all_docs),
        "total_chunks": len(chunks),
    })


@app.route("/ask_llama3", methods=["POST"])
def ask_llama3():
    json_content = request.json
    query = json_content.get("query")
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    # Adjusting the number of documents and similarity threshold for more precise retrieval
    vector_store = initialize_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        # Adjust these parameters as needed
        search_kwargs={"k": 10, "score_threshold": 0.5},
    )

    # Create a new chain with adjusted retriever
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    sources = [
        {"source": doc.metadata.get("source", "N/A"),
         "page_content": doc.page_content}
        for doc in result.get("context", [])
    ]

    return jsonify({"answer": result.get("answer", "No answer found"), "sources": sources})


def start_app(port=5000):
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    start_app(port=port)
