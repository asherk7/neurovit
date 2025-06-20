from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llm.llm_wrapper import LocalLLM
from rag.rag_utils import build_history_aware_retriever, build_document_chain
from langchain.schema import BaseRetriever
from langchain.chains.base import Chain

def build_rag_chain() -> Chain:
    """
    Builds a Retrieval-Augmented Generation (RAG) chain with a history-aware retriever and document
    combination chain, using a local LLM and a FAISS vector store of embeddings.

    Returns:
        Chain: A LangChain retrieval chain ready for question-answering with chat history.
    """

    # HuggingFace sentence-transformer embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"} 
    encode_kwargs = {"normalize_embeddings": False} 

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Load the local FAISS vector store from disk, allowing potentially unsafe deserialization
    vectorstore: BaseRetriever = FAISS.load_local(
        "rag/vectorstore", 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    
    llm = LocalLLM()

    # Build a history-aware retriever that combines vector search with chat history context
    history_aware_retriever: BaseRetriever = build_history_aware_retriever(
        vectorstore=vectorstore,
        llm=llm
    )

    # Build the document chain responsible for combining retrieved documents into a prompt
    document_chain: Chain = build_document_chain(llm=llm)
    
    # Create the full RAG chain by combining the retriever and the document combination chain
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=document_chain,
    )

    return rag_chain
