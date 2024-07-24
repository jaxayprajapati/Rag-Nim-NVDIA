import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS


# Load env variables

os.environ['NVIDIA_API_KEY'] = os.getenv('NVDIA_API_KEY')

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./ResearchPapers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("NVIDIA NIM Demo")

prompt = ChatPromptTemplate.from_template(
    """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        Questions:{input}
    """
)


single_text_prompt = st.text_input("Enter your Question From Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("FAISS Vectore Store DB Is ready Using NVIDIA Emebedding")

if single_text_prompt:
    # Create a document chain using the specified language model and prompt template
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    # Retrieve the vector store retriever from the session state
    retriever = st.session_state.vectors.as_retriever()
    
    # Create a retrieval chain combining the retriever and document chain
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    
    # Record the start time for processing
    start = time.process_time()
    
    # Invoke the retrieval chain with the user's input question and get the response
    response = retriever_chain.invoke({'input': single_text_prompt})
    
    # Print the response time to the console
    print("Response time: ", time.process_time() - start)
    
    # Display the answer from the response on the Streamlit app
    st.write(response['answer'])
    
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-"*100)









