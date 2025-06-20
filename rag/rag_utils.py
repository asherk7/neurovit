from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

def format_docs(docs: list[Document]) -> str:
    """
    Concatenates the page content from a list of Document objects into a single string separated by newlines.
    
    Args:
        docs (list[Document]): List of documents retrieved from the vector store.
    
    Returns:
        str: Concatenated document contents as plain text.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def build_document_chain(llm):
    """
    Builds a chain that combines documents into a prompt for the LLM to generate an answer.
    It uses a prompt template that includes instructions, chat history, the user input, 
    and relevant medical context.
    
    Args:
        llm: The local language model to use in the chain.
    
    Returns:
        Chain: A LangChain stuff_documents chain configured with the prompt and llm.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful medical assistant. Use the context below to answer clearly in plain text. Do not use markdown, bold, italic, or any formatting."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),        
        ("system", "Relevant medical info:\n{context}")
    ])

    # Create a stuff documents chain which concatenates all retrieved documents into the prompt and calls the llm
    return create_stuff_documents_chain(llm, prompt)


def build_history_aware_retriever(vectorstore, llm):
    """
    Builds a retriever that is aware of chat history, allowing the system to rewrite follow-up questions
    into standalone queries that consider the previous conversation.
    
    Args:
        vectorstore: The FAISS vectorstore or other retriever with document embeddings.
        llm: The local language model to use for rewriting follow-up questions.
    
    Returns:
        BaseRetriever: A history-aware retriever ready to be used in a RAG chain.
    """
    # Prompt template instructing how to rewrite a question using chat history
    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and a follow-up question, rewrite it to be a standalone question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create the history aware retriever, wrapping the vectorstore retriever with chat history rewriting capability
    return create_history_aware_retriever(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        prompt=retriever_prompt
    )
