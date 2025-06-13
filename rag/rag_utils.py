from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_document_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful medical assistant. Use the context below to answer clearly in plain text. Do not use markdown, bold, italic, or any formatting."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", "Relevant medical info:\n{context}")
    ])

    return create_stuff_documents_chain(llm, prompt)

def build_history_aware_retriever(vectorstore, llm):
    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and a follow-up question, rewrite it to be a standalone question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    return create_history_aware_retriever(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        prompt=retriever_prompt
    )
