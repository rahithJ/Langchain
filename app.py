import os
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = 'True'

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

llm = Ollama(model='llama3')
prompts = ChatPromptTemplate.from_messages(
    [
        ("system","You are the best AI Agent. Please response accordingly"),
        ("user","The Question {question}")
    ]
)

output_parser = StrOutputParser()

st.title("Rahith's First Chatbot")
input_txt = st.text_input("Whats on your Mind!!")


from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

document_chain = prompts|llm|output_parser

if input_txt:
    st.write(document_chain.invoke({"context":input_txt, "question":input_txt}))
