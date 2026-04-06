import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 현재 파일(ingest.py) 기준 상위 폴더의 data 폴더 경로 계산
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR) # src의 상위인 samsungfire 폴더

DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_PATH = os.path.join(DATA_DIR, "policy.pdf")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

def main():
    # 1. PDF 로드
    print(f"PDF 로드 중: {PDF_PATH}")
    if not os.path.exists(PDF_PATH):
        print("에러: PDF 파일이 존재하지 않습니다.")
        return
        
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"로드 완료: {len(documents)} 페이지")

    # 2. 텍스트 분할 (Chunking)
    # 한글 특성을 고려하여 충분한 chunk_size와 overlap 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"분할 완료: {len(chunks)} 청크")

    # 3. 임베딩 모델 설정
    # 한국어 성능이 우수한 모델 사용 (jhgan/ko-sroberta-multitask)
    print("임베딩 모델 로드 중 (최초 실행 시 시간이 걸릴 수 있습니다)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. 벡터 저장소(FAISS) 생성 및 저장
    print("벡터 데이터베이스 생성 중...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    print(f"FAISS 인덱스 저장 중: {INDEX_PATH}")
    vector_db.save_local(INDEX_PATH)
    print("저장 완료!")

if __name__ == "__main__":
    main()
