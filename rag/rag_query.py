from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llm.llm_wrapper import LocalLLM
from rag.rag_utils import build_history_aware_retriever, build_document_chain


def build_rag_chain():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vectorstore = FAISS.load_local(
        "rag/vectorstore", 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    
    llm = LocalLLM()

    history_aware_retriever = build_history_aware_retriever(vectorstore=vectorstore, llm=llm)
    document_chain = build_document_chain(llm=llm)
    
    rag_chain = create_retrieval_chain(
        retriever = history_aware_retriever,
        combine_docs_chain = document_chain
    )

    return rag_chain
