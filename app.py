import streamlit as st
import time
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


from dotenv import load_dotenv

load_dotenv()

st.title('Rag app')

urls = ['https://learntechnique.com/', 'https://www.elmtraining.com/resettlement-courses/','https://learntechnique.com/electrician-courses/am2-assessment/', 'https://learntechnique.com/contact/']
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)
all_splits = docs

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = OpenAI(temperature=0.4, max_tokens=500)

query = st.chat_input("Ask me anything regarding learntechnique: ")
prompt = query

system_prompt = ("""You are a helpful AI assistant that helps people find information about courses from the provided context.
If you don't know the answer, just say that you don't know. DO NOT try to make up an answer.
Use the following pieces of context to answer the question at the end.
{context}  
Answer the question truthfully and as best as you can and keep it concise.
""")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])


if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    response = rag_chain.invoke({"input": query})
    st.write(response['answer'])
