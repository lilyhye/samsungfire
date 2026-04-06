import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .env 파일 로드
load_dotenv()

class ChatbotEngine:
    """
    삼성화재용 RAG 엔진 클래스
    FAISS 벡터 데이터베이스를 로드하고 Groq API를 통해 답변을 생성합니다.
    사용자가 대시보드에서 직접 API 키를 입력할 수 있도록 설계되었습니다.
    """
    def __init__(self):
        # 1. 임베딩 모델 설정 (최초 실행 시 로드)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. FAISS 벡터 데이터베이스 로드 또는 자동 생성
        self.vector_db = self._prepare_vector_db()
        
        # 3. Groq 클라이언트 (초기에는 None, 나중에 세팅)
        self.client = None
        self._api_key = None

    def _prepare_vector_db(self):
        """인덱스가 있으면 로드하고, 없으면 생성합니다."""
        index_path = os.path.join("samsungfire", "data", "faiss_index")
        pdf_path = os.path.join("samsungfire", "data", "policy.pdf")
        
        if not os.path.exists(index_path):
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"원본 PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            vector_db = FAISS.from_documents(chunks, self.embeddings)
            vector_db.save_local(index_path)
            return vector_db
        else:
            return FAISS.load_local(
                index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )

    def set_api_key(self, api_key):
        """대시보드에서 입력받은 API 키를 설정하고 클라이언트를 초기화합니다."""
        if api_key and api_key != self._api_key:
            self._api_key = api_key
            self.client = Groq(api_key=api_key)
            return True
        return False

    def search_docs(self, query, k=3):
        """질문과 관련된 문서 조각을 검색합니다."""
        return self.vector_db.similarity_search(query, k=k)

    def get_streaming_response(self, query):
        """Groq API를 통해 스트리밍 답변을 생성합니다."""
        if not self.client:
            raise ValueError("API 키가 설정되지 않았습니다. 사이드바에서 키를 입력해 주세요.")

        # 1. 관련 문서 검색
        docs = self.search_docs(query)
        context = "\n".join([f"[문서 {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
        
        system_prompt = f"""당신은 삼성화재의 전문 상담 챗봇입니다. 
제공된 [참고 문서]의 내용을 바탕으로 고객의 질문에 정확하고 친절하게 답변해 주세요.

[참고 문서]
{context}

[답변 규칙]
1. 문서 내용에 기반하여 사실만 답변하세요.
2. 문서에 없는 내용인 경우 삼성화재 고객센터(1588-5114)로 안내하세요.
3. 한국어로 격식 있게 답변하세요.
"""

        try:
            return self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                stream=True,
            )
        except Exception as e:
            raise Exception(f"Groq API 호출 오류: {str(e)}")
