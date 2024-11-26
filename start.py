## streamlit 관련 모듈 불러오기
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re

## 환경변수 불러오기
from dotenv import load_dotenv
load_dotenv()


import streamlit as st

# Streamlit Secrets에서 API 키 가져오기
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: 여러 파일을 임시 폴더에 저장
def save_uploadedfiles(uploaded_files: List[UploadedFile]) -> List[str]:
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        file_paths.append(file_path)
    return file_paths

## 2: 저장된 PDF 파일들을 Document로 변환
def pdfs_to_documents(pdf_paths: List[str]) -> List[Document]:
    documents = []
    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(pdf_path)
        doc = loader.load()
        for d in doc:
            d.metadata['file_path'] = pdf_path
        documents.extend(doc)
    return documents

## 3: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

## 4: Document를 벡터DB로 저장
def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")


############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################

## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question: str):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    # 벡터 DB 호출
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    # 사용자 질문을 기반으로 관련 문서 검색
    retrieve_docs: List[Document] = retriever.invoke(user_question)
    # RAG 체인 선언
    chain = get_rag_chain()
    # 질문과 문맥을 넣어서 체인 결과 호출
    response = chain.invoke({"question": user_question, "context": retrieve_docs})
    return response, retrieve_docs


def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘:
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    컨텍스트: {context}
    질문: {question}
    응답:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    return custom_rag_prompt | model | StrOutputParser()


############################### 3단계 : 응답결과와 문서를 함께 보도록 도와주는 함수 ##########################
@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)  # 문서 열기
    image_paths = []
    output_folder = "PDF_이미지"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)  # type: ignore
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    return image_paths


def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]


############################### 메인 함수 ##########################
def main():
    st.set_page_config("DWS QnA 챗봇", layout="wide")

    left_column, right_column = st.columns([1, 1])
    with left_column:
        st.header("DWS QnA 챗봇")

        uploaded_files = st.file_uploader("PDF 파일 업로드", type="pdf", accept_multiple_files=True)
        button = st.button("PDF 업로드하기")

        if uploaded_files and button:
            with st.spinner("PDF 문서 처리 중"):
                pdf_paths = save_uploadedfiles(uploaded_files)
                pdf_documents = pdfs_to_documents(pdf_paths)
                smaller_documents = chunk_documents(pdf_documents)
                save_to_vector_store(smaller_documents)

        user_question = st.text_input("PDF 문서에 대해서 질문해 주세요",
                                       placeholder="검사업무는 누가 하나요?")

        if user_question:
            response, context = process_question(user_question)
            st.write(response)
            for i, document in enumerate(context):
                with st.expander(f"관련 문서 {i + 1}"):
                    st.write(document.page_content)
                    file_path = document.metadata.get('file_path', '')
                    page_number = document.metadata.get('page', 0) + 1
                    st.write(f"파일: {os.path.basename(file_path)} | 페이지: {page_number}")


if __name__ == "__main__":
    main()
