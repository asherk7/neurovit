from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents.base import Document
from llm.llm_wrapper import LocalLLM 

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
    
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful medical assistant. Use the context below to answer clearly in plain text. Do not use markdown, bold, italic, or any formatting."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        ("system", "Relevant medical info:\n{context}")
    ])

    llm = LocalLLM()

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)