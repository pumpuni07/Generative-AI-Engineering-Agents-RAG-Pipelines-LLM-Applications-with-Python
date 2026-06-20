"""
Lab 7: QA Bot — LangChain + IBM watsonx.ai + Gradio
=====================================================
Topic: A full end-to-end Retrieval-Augmented Generation (RAG) QA bot
       that answers questions from uploaded PDF documents.

Pipeline (6 components):
  1. Document Loader   — PyPDFLoader reads a PDF into LangChain Documents
  2. Text Splitter     — RecursiveCharacterTextSplitter chunks the text
  3. Embedding Model   — WatsonxEmbeddings converts chunks to dense vectors
  4. Vector Database   — ChromaDB stores and indexes the embeddings
  5. Retriever         — Similarity search fetches the most relevant chunks
  6. QA Chain          — RetrievalQA passes context + query to the LLM

Front-end:
  - Gradio gr.Interface with PDF file upload + text input + text output

All 6 tasks from the lab are fully implemented here:
  Task 1: document_loader()    — PyPDFLoader
  Task 2: text_splitter()      — RecursiveCharacterTextSplitter
  Task 3: watsonx_embedding()  — IBM Slate 125M embeddings
  Task 4: vector_database()    — ChromaDB vector store
  Task 5: retriever()          — vector store similarity retriever
  Task 6: retriever_qa()       — RetrievalQA chain + Gradio interface

Prerequisites:
  pip install gradio==4.44.0 ibm-watsonx-ai==1.1.2 langchain==0.2.11 \\
              langchain-community==0.2.10 langchain-ibm==0.1.11 \\
              chromadb==0.4.24 pypdf==4.3.1 pydantic==2.9.1 \\
              huggingface_hub==0.23.0

Usage:
  python lab7_qabot_pdf_rag.py
  Then open http://127.0.0.1:7860 and upload a PDF to start asking questions.

Note on PDF size:
  Very large PDFs will be slow or may fail due to memory / API limits.
  For best results, use PDFs under ~50 pages.

Author notes:
  Based on IBM Skills Network lab material (Kang Wang, IBM / U. Waterloo;
  Joseph Santarcangelo, IBM; Hailey Quach, IBM / Concordia U.).
  Fully implemented (all blanks filled) and extended with detailed
  explanations by Jack Pumpuni Frimpong-Manso.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings("ignore")

# LangChain + IBM watsonx.ai
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from huggingface_hub import HfFolder

# Gradio
import gradio as gr


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
WATSONX_URL        = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID         = "skills-network"   # Free in IBM Cloud IDE

LLM_MODEL_ID       = "ibm/granite-4-h-small"
EMBED_MODEL_ID     = "ibm/slate-125m-english-rtrvr"

LLM_MAX_TOKENS     = 256
LLM_TEMPERATURE    = 0.5

CHUNK_SIZE         = 1000
CHUNK_OVERLAP      = 200

SERVER_NAME        = "0.0.0.0"   # Use "127.0.0.1" for local-only access
SERVER_PORT        = 7860


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: DOCUMENT LOADER
# ─────────────────────────────────────────────────────────────────────────────
def document_loader(file_path: str) -> list:
    """
    Task 1 — Load a PDF document using LangChain's PyPDFLoader.

    PyPDFLoader reads each page of the PDF into a LangChain Document object.
    Each Document contains:
      - page_content (str): The extracted text of the page.
      - metadata (dict):    Source filename and page number.

    Note:
      PyPDFLoader's .load() method returns a flat list of Documents
      (one per page). It does NOT split text into chunks — that is
      handled separately by text_splitter() in Task 2.

    Args:
        file_path (str): Path to the uploaded PDF file.

    Returns:
        list[Document]: List of LangChain Document objects (one per page).
    """
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    return loaded_document


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: TEXT SPLITTER
# ─────────────────────────────────────────────────────────────────────────────
def text_splitter(data: list) -> list:
    """
    Task 2 — Split loaded document pages into smaller chunks.

    Why split?
      LLMs have a context window limit. Long documents cannot be fed
      into the LLM at once. Splitting creates smaller, semantically
      coherent chunks that can each fit within the context window.

    RecursiveCharacterTextSplitter strategy:
      Tries to split on paragraph boundaries (\\n\\n), then sentences (\\n),
      then words (" "), then characters ("") — in that priority order.
      This preserves semantic coherence better than fixed-character splits.

    Parameters:
      chunk_size=1000    → Maximum characters per chunk.
      chunk_overlap=200  → Characters shared between adjacent chunks,
                           ensuring context is not lost at boundaries.
      length_function=len → Standard character-count length function.

    Args:
        data (list[Document]): Output of document_loader().

    Returns:
        list[Document]: List of smaller Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────────────────────
def watsonx_embedding() -> WatsonxEmbeddings:
    """
    Task 3 — Initialise IBM watsonx.ai embedding model.

    Uses IBM's Slate 125M English embeddings model to convert text chunks
    into dense vector representations (embeddings).

    Why embeddings?
      Embeddings map text into a high-dimensional vector space where
      semantically similar texts are geometrically close. This enables
      efficient similarity search in the vector database.

    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS:
      Truncates input text that exceeds the model's maximum token limit.
      Set to 3 to instruct the model to process up to 3 tokens (lab default).

    Returns:
        WatsonxEmbeddings: LangChain-compatible embedding model instance.
    """
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    embedding_model = WatsonxEmbeddings(
        model_id=EMBED_MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=embed_params,
    )
    return embedding_model


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: VECTOR DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def vector_database(chunks: list) -> Chroma:
    """
    Task 4 — Create a ChromaDB vector store from document chunks.

    Process:
      1. Each chunk's text is passed to the embedding model.
      2. The resulting embedding vector is stored in ChromaDB alongside
         the original text and metadata.
      3. ChromaDB builds an index enabling fast approximate nearest-neighbour
         (ANN) search.

    ChromaDB is an open-source, in-memory vector database that is well-suited
    for prototyping RAG systems. For production, consider Pinecone, Weaviate,
    or FAISS.

    Args:
        chunks (list[Document]): Output of text_splitter().

    Returns:
        Chroma: A ChromaDB vector store containing all chunk embeddings.
    """
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5: RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────
def retriever(file_path: str):
    """
    Task 5 — Build a vector store-based retriever for a given PDF file.

    This function orchestrates Tasks 1–4 into a single pipeline:
      PDF → load → split → embed → store → retriever

    Retriever type: similarity search (cosine similarity by default in Chroma).
      The retriever accepts a query string and returns the top-k most
      semantically similar chunks from the vector store.

    Alternative retriever types (not used here but mentioned in lab):
      - MMR (Maximum Marginal Relevance): balances relevance and diversity.
      - Parent Document Retriever: fetches broader context around a chunk.
      - Self-Query Retriever: parses structured filters from natural language.

    Args:
        file_path (str): Path to the PDF file to process.

    Returns:
        VectorStoreRetriever: A LangChain retriever ready for use in a chain.
    """
    # Task 1: Load
    splits = document_loader(file_path)
    # Task 2: Split
    chunks = text_splitter(splits)
    # Tasks 3 & 4: Embed + store
    vectordb = vector_database(chunks)
    # Task 5: Build retriever from vector store
    retriever_obj = vectordb.as_retriever()
    return retriever_obj


# ─────────────────────────────────────────────────────────────────────────────
# LLM INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def get_llm() -> WatsonxLLM:
    """
    Initialises the IBM watsonx.ai LLM (Granite 4H Small by default).

    Configuration:
      MAX_NEW_TOKENS = 256 : Maximum response length. Increase if truncated.
      TEMPERATURE    = 0.5 : Balanced creativity / accuracy for Q&A.

    Returns:
        WatsonxLLM: LangChain-compatible LLM wrapper.
    """
    parameters = {
        GenParams.MAX_NEW_TOKENS: LLM_MAX_TOKENS,
        GenParams.TEMPERATURE:    LLM_TEMPERATURE,
    }
    watsonx_llm = WatsonxLLM(
        model_id=LLM_MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=parameters,
    )
    return watsonx_llm


# ─────────────────────────────────────────────────────────────────────────────
# TASK 6: QA CHAIN
# ─────────────────────────────────────────────────────────────────────────────
def retriever_qa(file, query: str) -> str:
    """
    Task 6 — Full RAG question-answering pipeline.

    How RetrievalQA works:
      1. The retriever fetches the most relevant document chunks for `query`.
      2. LangChain formats a prompt: [context chunks] + [user question].
      3. The LLM reads the combined prompt and generates an answer grounded
         in the retrieved context — not just its pretrained knowledge.
      4. return_source_documents=False: only return the answer string,
         not the source chunks (set True to inspect which chunks were used).

    chain_type="stuff":
      "Stuff" is the simplest chain type — it stuffs all retrieved chunks
      into a single prompt. Suitable for small-to-medium PDFs.
      Alternative chain types for large documents:
        - "map_reduce": processes chunks independently then combines answers.
        - "refine":     iteratively refines the answer with each chunk.
        - "map_rerank": ranks answers from each chunk and picks the best.

    Args:
        file:       Gradio file object with a .name attribute (temp file path).
        query (str): The user's natural language question.

    Returns:
        str: The LLM-generated answer grounded in the PDF content.
    """
    # Guard: handle both Gradio file objects and plain file paths
    file_path = file.name if hasattr(file, "name") else file

    if not file_path:
        return "⚠️  Please upload a PDF file first."
    if not query.strip():
        return "⚠️  Please enter a question."

    try:
        llm          = get_llm()
        retriever_obj = retriever(file_path)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",                 # Task 6 blank: chain_type
            retriever=retriever_obj,            # Task 6 blank: retriever
            return_source_documents=False,      # Task 6 blank: return_source_documents
        )

        response = qa.invoke({"query": query})  # Task 6 blank: invoke argument
        return response["result"]

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
def build_gradio_interface() -> gr.Interface:
    """
    Constructs the Gradio web interface for the QA bot.

    Components:
      - gr.File      : PDF upload (single file, .pdf extension only)
      - gr.Textbox   : User's question input (2-line expandable)
      - gr.Textbox   : Bot's answer output

    allow_flagging="never":
      Disables the Gradio flagging button (not needed for this use case).

    Returns:
        gr.Interface: Configured but not yet launched Gradio app.
    """
    rag_application = gr.Interface(
        fn=retriever_qa,
        allow_flagging="never",
        inputs=[
            gr.File(
                label="📄 Upload PDF File",
                file_count="single",
                file_types=[".pdf"],
                type="filepath",
            ),
            gr.Textbox(
                label="❓ Input Query",
                lines=2,
                placeholder="Type your question here...",
            ),
        ],
        outputs=gr.Textbox(
            label="💬 Answer",
            lines=6,
        ),
        title="📚 PDF Question Answering Bot",
        description=(
            "Upload a PDF document and ask any question about its content. "
            "The bot uses IBM watsonx.ai and LangChain RAG to answer based "
            "on the document — not just general knowledge.\n\n"
            "**Best results:** PDFs under 50 pages. Scanned image-only PDFs "
            "are not supported (text must be extractable)."
        ),
        examples=[
            # Examples require a local PDF — left as placeholders
            # ["/path/to/sample.pdf", "What is the main topic of this document?"],
        ],
    )
    return rag_application


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Launching PDF QA Bot …")
    print(f"   LLM     : {LLM_MODEL_ID}")
    print(f"   Embedder: {EMBED_MODEL_ID}")
    print(f"   Chunk   : size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"   Server  : http://{SERVER_NAME}:{SERVER_PORT}\n")

    app = build_gradio_interface()
    app.launch(server_name=SERVER_NAME, server_port=SERVER_PORT)
