
# import streamlit as st
# import os
# import time
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


# # load_dotenv()
# # os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
# # os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
# # groq_api_key = os.getenv("GROQ_API_KEY")

# def get_api_keys():
#     # First try to load from .env file (local development)
#     load_dotenv()
#     groq_key = os.getenv("GROQ_API_KEY")
    
#     # If not found in .env, try Streamlit Secrets (cloud deployment)
#     if groq_key is None:
#         try:
#             groq_key = st.secrets["GROQ_API_KEY"]
#         except Exception:
#             pass
    
#     # If still not found, show error
#     if groq_key is None:
#         st.error("""
#         GROQ API key not found. Please ensure one of the following:
#         1. A .env file exists in your project root with GROQ_API_KEY
#         2. Streamlit Secrets are configured (for cloud deployment)
#         """)
#         st.stop()
    
#     return groq_key

# groq_api_key = get_api_keys()

# llm = ChatGroq(
#     groq_api_key=groq_api_key,
#     model_name="Llama3-8b-8192"
# )

# # Custom CSS styling
# st.markdown("""
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
#         .main-title {
#             font-family: 'Roboto', sans-serif;
#             font-size: 70px;
#             font-weight: 700;
#             color: #1E88E5;
#             margin-bottom: 20px;
#         }
        
#         .response-container {
#             background-color: #f0f2f6;
#             border-radius: 10px;
#             padding: 20px;
#             margin: 10px 0;
#             font-family: 'Roboto', sans-serif;
#             font-size: 16px;
#             line-height: 1.6;
#         }
        
#         .similar-doc {
#             background-color: white;
#             border-left: 4px solid #1E88E5;
#             padding: 15px;
#             margin: 10px 0;
#             font-family: 'Roboto', sans-serif;
#             font-size: 14px;
#         }
        
#         .metric-container {
#             background-color: #e3f2fd;
#             padding: 10px;
#             border-radius: 5px;
#             margin: 5px 0;
#             font-family: 'Roboto', sans-serif;
#             font-size: 14px;
#         }

        
#     </style>
# """, unsafe_allow_html=True)


# prompt = ChatPromptTemplate.from_template("""
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question
    
#     <context>
#     {context}
#     </context>
    
#     Question: {input}
# """)

# def create_vector_embedding():
#     """Create vector embeddings if they don't exist in session state"""
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = OpenAIEmbeddings()
#         # Data Ingestion step
#         st.session_state.loader = PyPDFDirectoryLoader("research_papers")
#         # Document Loading
#         st.session_state.docs = st.session_state.loader.load()
        
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
        
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(
#             st.session_state.docs[:50]
#         )
        
#         st.session_state.vectors = FAISS.from_documents(
#             st.session_state.final_documents,
#             st.session_state.embeddings
#         )

# # Streamlit UI - Updated Header Section
# st.title("AI powered Banking Assistance")
# # st.markdown('<p class="main-title">AI powered Banking Assistance</p>', unsafe_allow_html=True)
# st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)
# st.markdown("""
#     <p class="subtitle">
#         Your 24/7 Digital Banking Guide for Questions About Accounts, Cards, and Services
#     </p>
# """, unsafe_allow_html=True)

# # Add banking-specific features section
# st.markdown("""
#     <div style='display: flex; justify-content: center; gap: 20px; margin: 30px 0; font-family: Roboto, sans-serif;'>
#         <div class='feature-box'>
#             💳 Credit & Debit Cards
#         </div>
#         <div class='feature-box'>
#             💰 Savings Accounts
#         </div>
#         <div class='feature-box'>
#             🏦 Banking Services
#         </div>
#     </div>
# """, unsafe_allow_html=True)


# user_prompt = st.text_input(
#     "How can I help you with your banking queries today?",
#     help="Ask about savings accounts, credit cards, debit cards, or any other banking services"
# )

# if st.button("Initialize Banking Knowledge", type="primary"):
#     with st.spinner("Setting up banking information..."):
#         create_vector_embedding()
#     st.success("✔️ Banking assistant is ready to help")

# if user_prompt:
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
#     with st.spinner("Finding the best answer for you..."):
#         start = time.process_time()
#         response = retrieval_chain.invoke({'input': user_prompt})
#         response_time = time.process_time() - start
    
#     # Display response time metric
#     # st.markdown(f"""
#     #     <div class="metric-container">
#     #         ⚡ Response Time: {response_time:.2f} seconds
#     #     </div>
#     # """, unsafe_allow_html=True)
    
#     # Display the main response
#     st.markdown(f"""
#         <div class="response-container">
#             {response['answer']}
#         </div>
#     """, unsafe_allow_html=True)
    
#     # Display relevant banking information
#     with st.expander("📑 Related Banking Information"):
#         for i, doc in enumerate(response['context']):
#             st.markdown(f"""
#                 <div class="similar-doc">
#                     {doc.page_content}
#                 </div>
#             """, unsafe_allow_html=True)







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

# API key management
def get_api_keys():
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    
    if groq_key is None:
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    
    if groq_key is None:
        st.error("""
        GROQ API key not found. Please ensure one of the following:
        1. A .env file exists in your project root with GROQ_API_KEY
        2. Streamlit Secrets are configured (for cloud deployment)
        """)
        st.stop()
    
    return groq_key

groq_api_key = get_api_keys()
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Custom CSS styling
st.markdown("""
    <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        /* Global styles */
        :root {
            --primary-color: #1E88E5;
            --text-color: #2C3E50;
            --bg-color: #FFFFFF;
            --container-bg: #f0f2f6;
            --border-color: #E0E0E0;
        }

        [data-testid="stAppViewContainer"] {
            background-color: var(--bg-color);
            color: var(--text-color);
        }

        /* Dark mode */
        @media (prefers-color-scheme: dark) {
            :root {
                --primary-color: #90CAF9;
                --text-color: #FFFFFF;
                --bg-color: #1A1A1A;
                --container-bg: #2D2D2D;
                --border-color: #404040;
            }
        }

        /* Title styles */
        .main-title {
            font-family: 'Roboto', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-color);
            text-align: center;
            margin: 2rem 0;
            padding: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        /* Subtitle styles */
        .subtitle {
            font-family: 'Roboto', sans-serif;
            font-size: 1.2rem;
            color: var(--text-color);
            text-align: center;
            margin-bottom: 2rem;
            opacity: 0.9;
        }

        /* Feature box styles */
        .feature-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 1rem;
            margin: 2rem 0;
        }

        .feature-box {
            background-color: var(--container-bg);
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 200px;
            max-width: 300px;
            text-align: center;
            transition: transform 0.2s;
        }

        .feature-box:hover {
            transform: translateY(-2px);
        }

        /* Response container */
        .response-container {
            background-color: var(--container-bg);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid var(--border-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            color: var(--text-color);
        }

        /* Similar document styles */
        .similar-doc {
            background-color: var(--container-bg);
            border-left: 4px solid var(--primary-color);
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
            color: var(--text-color);
        }

        /* Input field enhancement */
        .stTextInput > div > div {
            background-color: var(--container-bg);
            color: var(--text-color);
            border-radius: 8px;
        }

        /* Button enhancement */
        .stButton > button {
            background-color: var(--primary-color);
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--container-bg);
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Chat prompt template
prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    
    <context>
    {context}
    </context>
    
    Question: {input}
""")

# Vector embedding creation
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

# App Header
st.markdown('<h1 class="main-title">AI-Powered Banking Assistant</h1>', unsafe_allow_html=True)
st.markdown("""
    <p class="subtitle">
        Your 24/7 Digital Banking Guide for Questions About Accounts, Cards, and Services
    </p>
""", unsafe_allow_html=True)

# Feature Boxes
st.markdown("""
    <div class="feature-container">
        <div class="feature-box">
            <div style="font-size: 2rem;">💳</div>
            <div style="font-weight: 500;">Credit & Debit Cards</div>
        </div>
        <div class="feature-box">
            <div style="font-size: 2rem;">💰</div>
            <div style="font-weight: 500;">Savings Accounts</div>
        </div>
        <div class="feature-box">
            <div style="font-size: 2rem;">🏦</div>
            <div style="font-weight: 500;">Banking Services</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# User Input
user_prompt = st.text_input(
    "How can I help you with your banking queries today?",
    help="Ask about savings accounts, credit cards, debit cards, or any other banking services"
)

# Initialize Button
if st.button("Initialize Banking Knowledge", type="primary"):
    with st.spinner("Setting up banking information..."):
        create_vector_embedding()
    st.success("✔️ Banking assistant is ready to help")

# Process Query and Display Response
if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    with st.spinner("Finding the best answer for you..."):
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        response_time = time.process_time() - start
    
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
