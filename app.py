import streamlit as st
import os
from chatbot_engine import ChatbotEngine
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="삼성화재 RAG 고객 상담 챗봇",
    page_icon="🛡️",
    layout="centered"
)

# --- 경로 진단 및 디버깅 (배포 시 필수 확인용) ---
with st.sidebar:
    with st.expander("🔍 [디버그] 현재 서버 경로 및 파일 상태"):
        st.write(f"현재 작업 디렉토리: `{os.getcwd()}`")
        st.write(f"__file__ 경로: `{__file__}`")
        # 프로젝트 루트 파일 목록 확인 (최대 2단계 깊이)
        root_files = []
        for root, dirs, files in os.walk(".", topdown=True):
            level = root.replace(".", "").count(os.sep)
            if level <= 1:
                for f in files:
                    root_files.append(os.path.join(root, f))
        st.write("주요 파일 목록:")
        st.code("\n".join(root_files[:20]))

st.title("🛡️ 삼성화재 RAG 고객 상담 챗봇")
st.markdown("""
이 챗봇은 삼성화재의 공식 문서를 바탕으로 답변을 제공합니다.  
**사이드바에 Groq API 키를 입력**하신 후 대화를 시작해 보세요.
""")

# --- 사이드바 설정 ---
with st.sidebar:
    st.image("https://www.samsungfire.com/common/img/logo.png", width=150)
    st.divider()
    
    st.subheader("🔑 API 키 설정")
    # API 키 입력창
    user_api_key = st.text_input(
        "Groq API Key를 입력하세요", 
        type="password",
        placeholder="gsk_...",
        value=os.getenv("GROQ_API_KEY", "") # .env에 있으면 기본값으로 채움
    )
    
    if user_api_key:
        st.success("API 키가 입력되었습니다.")
    else:
        st.warning("API 키를 입력해야 서비스 이용이 가능합니다.")
        st.caption("[Groq Console](https://console.groq.com/keys)에서 키를 발급받을 수 있습니다.")

    st.divider()
    st.subheader("⚙️ 모델 정보")
    st.info("LLM: Llama 3.1 8B (Groq)\n\nEmbeddings: ko-sroberta-multitask")

# --- 챗봇 로직 ---

# 1. 챗봇 엔진 초기화 (캐싱)
@st.cache_resource
def load_chatbot():
    try:
        # 엔진 객체 생성 (임베딩 및 FAISS 로드 - 이 과정은 API 키 없이도 수행됨)
        return ChatbotEngine()
    except Exception as e:
        st.error(f"엔진 초기화 실패: {e}")
        return None

# 엔진 로드
# 스피너는 최초 실행 시(임베딩 로드)만 표시됨
chatbot = load_chatbot()

# API 키 동적 반영
if chatbot and user_api_key:
    chatbot.set_api_key(user_api_key)

# 2. 채팅 인터페이스 (세션 상태)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 사용자 입력 처리
# API 키가 있을 때만 입력창 활성화
input_disabled = not user_api_key
if prompt := st.chat_input("질문을 입력해 주세요", disabled=input_disabled):
    # 사용자 메시지 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 챗봇 응답 생성
    if chatbot and user_api_key:
        with st.chat_message("assistant"):
            try:
                with st.expander("🔍 관련 근거 확인"):
                    relevant_docs = chatbot.search_docs(prompt)
                    for i, doc in enumerate(relevant_docs):
                        st.caption(f"**[참고 {i+1}]** {doc.page_content[:200]}...")
                
                # 스트리밍 응답 컨테이너
                message_placeholder = st.empty()
                full_response = ""
                
                # 엔진을 통해 답변 생성
                completion = chatbot.get_streaming_response(prompt)
                
                if completion:
                    for chunk in completion:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
    elif input_disabled:
        st.info("회의를 시작하려면 사이드바에 API 키를 입력해 주세요.")
