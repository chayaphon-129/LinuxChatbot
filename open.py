import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


st.set_page_config(page_title="Linux/Unix Command Chatbot 🤖", layout="centered") # ตั้งค่าหน้าเพจ

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# 1. Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data_path):
    """Loads PDF files from a directory."""
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

documents = load_pdf_files(DATA_PATH)

# 2. Create Chunks
def create_chunks(documents):
    """Splits documents into smaller text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
    text_chunks = text_splitter.split_documents(documents)
    print(f"Created {len(text_chunks)} text chunks.")
    return text_chunks

text_chunks = create_chunks(documents)

# 3. Create Vector Embeddings
def get_embedding_model():
    """Initializes the OpenAI embedding model."""
    embedding_model = OpenAIEmbeddings()
    return embedding_model

embedding_model = get_embedding_model()

# 4. จัดเก็บ Embeddings ใน FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def create_vectorstore(_text_chunks, _embedding_model, db_faiss_path):
    """Creates and saves the FAISS vectorstore."""
    db = FAISS.from_documents(_text_chunks, _embedding_model)
    db.save_local(db_faiss_path)
    print(f"Saved FAISS index to {db_faiss_path}")
    return db

db = create_vectorstore(text_chunks, embedding_model, DB_FAISS_PATH)

# 5. ตั้งค่า LLM สำหรับ QA
def get_llm():
    """Initializes the OpenAI LLM."""
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5) 
    return llm

llm = get_llm()

# 6. สร้าง RetrievalQA Chain
def create_qa_chain(llm, db):
    """Creates the RetrievalQA chain."""
    prompt_template = """ใช้ข้อมูลที่ส่งให้ในการตอบคำถามเท่านั้น.
    ถ้าคุณไม่รู้คำตอบ, คุณตอบแค่เพียงว่า ขอโทษครับผมไม่ทราบ. อย่าสร้างคำตอบขึ้นมาเอง.

    Context: {context}
    Question: {question}

    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}), 
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

qa_chain = create_qa_chain(llm, db)

# 7. Chat with the LLM using Streamlit
def chat_with_llm():
    """Starts the chat interaction with the LLM using Streamlit."""
    #st.set_page_config(page_title="Linux/Unix Command Chatbot 🤖", layout="centered") # ตั้งค่าหน้าเพจ
    st.title("Command Line LinuxChatbot")
    st.markdown("ยินดีต้อนรับ! 👋 ถามคำถามเกี่ยวกับ **คำสั่ง Linux และ Unix** ได้ที่นี่")

    # Sidebar สำหรับข้อมูลเพิ่มเติม
    with st.sidebar:
        st.header("เกี่ยวกับ Chatbot นี้")
        st.info(
            "Chatbot นี้ถูกสร้างขึ้นเพื่อตอบคำถามเกี่ยวกับคำสั่ง Linux และ Unix "
            "โดยอ้างอิงจากเอกสาร PDF ใน LLM."
        )
        st.markdown("**วิธีใช้งาน:**")
        st.markdown("- พิมพ์คำถามของคุณในช่องด้านล่าง.")
        st.markdown("- ระบบจะค้นหาคำตอบจากเอกสารและแสดงผลให้คุณทราบ.")
        st.markdown("- หากมีเอกสารที่อ้างอิง จะแสดงในส่วน 'เอกสารอ้างอิง'.")

    # เริ่มต้นสถานะการสนทนา
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # แสดงข้อความแชทที่มีอยู่
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ช่องรับคำถามจากผู้ใช้
    if prompt := st.chat_input("ถามเกี่ยวกับ Linux คำสั่งยูนิกซ์/ลินุกซ์:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # เรียกใช้ QA Chain
        try:
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # สร้างข้อความสำหรับแสดงผล
            full_response_content = result

            st.session_state.messages.append({"role": "assistant", "content": full_response_content})
            with st.chat_message("assistant"):
                st.markdown(full_response_content)

                # แสดงเอกสารอ้างอิงแยกต่างหากใน expander
                if source_documents:
                    with st.expander("🔗 เอกสารอ้างอิง"):
                        for i, doc in enumerate(source_documents):
                            st.markdown(f"**เอกสารที่ {i+1}**: {doc.metadata.get('source', 'ไม่ระบุ')}")
                            st.code(doc.page_content, language='text')
                            st.markdown("---") # เส้นแบ่งเอกสาร
        except Exception as e:
            error_message = f"เกิดข้อผิดพลาด: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    chat_with_llm()

    
