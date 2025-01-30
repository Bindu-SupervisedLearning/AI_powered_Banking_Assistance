# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# import openai

# from dotenv import load_dotenv
# load_dotenv()
# ## load the GROQ API Key
# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
# os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

# groq_api_key=os.getenv("GROQ_API_KEY")

# llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

# prompt=ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate respone based on the question
#     <context>
#     {context}
#     <context>
#     Question:{input}

#     """

# )

# def create_vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings=OpenAIEmbeddings()
#         st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## Data Ingestion step
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
# st.title("Banking Assistant AI")

# # user_prompt=st.text_input("Enter your query from the research paper")

# user_prompt = st.text_input(
#     "How can I help you with your banking queries today?",
#     help="Ask about savings accounts, credit cards, debit cards, or any other banking services"
# )

# if st.button("Document Embedding"):
#     create_vector_embedding()
#     st.write("Vector Database is ready")

# import time

# if user_prompt:
#     document_chain=create_stuff_documents_chain(llm,prompt)
#     retriever=st.session_state.vectors.as_retriever()
#     retrieval_chain=create_retrieval_chain(retriever,document_chain)

#     start=time.process_time()
#     response=retrieval_chain.invoke({'input':user_prompt})
#     print(f"Response time :{time.process_time()-start}")

#     st.write(response['answer'])

#     with st.expander("Document similarity Search"):
#         for i,doc in enumerate(response['context']):
#             st.write(doc.page_content)
#             st.write('------------------------')




import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai
from dotenv import load_dotenv


load_dotenv()
# os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Custom CSS styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
        .main-title {
            font-family: 'Roboto', sans-serif;
            font-size: 40px;
            font-weight: 700;
            color: #1E88E5;
            margin-bottom: 20px;
        }
        
        .response-container {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
            line-height: 1.6;
        }
        
        .similar-doc {
            background-color: white;
            border-left: 4px solid #1E88E5;
            padding: 15px;
            margin: 10px 0;
            font-family: 'Roboto', sans-serif;
            font-size: 14px;
        }
        
        .metric-container {
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            font-family: 'Roboto', sans-serif;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)


prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    
    <context>
    {context}
    </context>
    
    Question: {input}
""")

def create_vector_embedding():
    """Create vector embeddings if they don't exist in session state"""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        # Data Ingestion step
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        # Document Loading
        st.session_state.docs = st.session_state.loader.load()
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )
        
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

# Streamlit UI - Updated Header Section
st.markdown('<p class="main-title">AI powered Banking Assistance</p>', unsafe_allow_html=True)
st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)
st.markdown("""
    <p class="subtitle">
        Your 24/7 Digital Banking Guide for Questions About Accounts, Cards, and Services
    </p>
""", unsafe_allow_html=True)

# Add banking-specific features section
st.markdown("""
    <div style='display: flex; justify-content: center; gap: 20px; margin: 30px 0; font-family: Roboto, sans-serif;'>
        <div class='feature-box'>
            💳 Credit & Debit Cards
        </div>
        <div class='feature-box'>
            💰 Savings Accounts
        </div>
        <div class='feature-box'>
            🏦 Banking Services
        </div>
    </div>
""", unsafe_allow_html=True)

# st.title("AI powered Banking Assistance")
user_prompt = st.text_input(
    "How can I help you with your banking queries today?",
    help="Ask about savings accounts, credit cards, debit cards, or any other banking services"
)

if st.button("Initialize Banking Knowledge", type="primary"):
    with st.spinner("Setting up banking information..."):
        create_vector_embedding()
    st.success("✔️ Banking assistant is ready to help")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    with st.spinner("Finding the best answer for you..."):
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        response_time = time.process_time() - start
    
    # Display response time metric
    # st.markdown(f"""
    #     <div class="metric-container">
    #         ⚡ Response Time: {response_time:.2f} seconds
    #     </div>
    # """, unsafe_allow_html=True)
    
    # Display the main response
    st.markdown(f"""
        <div class="response-container">
            {response['answer']}
        </div>
    """, unsafe_allow_html=True)
    
    # Display relevant banking information
    with st.expander("📑 Related Banking Information"):
        for i, doc in enumerate(response['context']):
            st.markdown(f"""
                <div class="similar-doc">
                    {doc.page_content}
                </div>
            """, unsafe_allow_html=True)
