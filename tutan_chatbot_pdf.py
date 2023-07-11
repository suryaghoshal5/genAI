import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'streamlit_chat'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'langchain'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tempfile'])


import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import tempfile
import os 

os.environ["OPENAI_API_KEY"] = "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"

uploaded_file = st.sidebar.file_uploader("upload", type="pdf")
if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    #loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
    loader = PyPDFLoader(file_path=tmp_file_path)
    pages = loader.load_and_split()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    #print(f"Loaded {len(docs)} documents from {file['name']}")

    data = loader.load()
    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)
    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.4,model_name='gpt-3.5-turbo-0613', 
        openai_api_key="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"),
        retriever=vectors.as_retriever())
    def conversational_chat(query):
        
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk about your pdf file here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                
#streamlit run tuto_chatbot_csv.py

