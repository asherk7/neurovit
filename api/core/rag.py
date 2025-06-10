#langchain pipeline + vectorstore

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load vectorstore (already populated with medical docs)
db = FAISS.load_local("path_to_vectorstore", HuggingFaceEmbeddings())

# Load LLM
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("...", torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = HuggingFacePipeline(pipeline=pipe)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

def answer_question(question: str, tumor_type: str):
    query = f"{tumor_type} tumor: {question}"
    result = qa.run(query)
    return result, ["source1", "source2"]  # Modify with actual sources
