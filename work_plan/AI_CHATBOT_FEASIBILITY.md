# 🤖 AI Context-Aware Chatbot Feasibility Review

## 1. 결론: 구현 가능 (Highly Feasible) ✅

사용자가 요청한 **"모든 탭의 내용을 이해하고 처리하는 사이드바 AI 챗봇"**은 기술적으로 충분히 구현 가능하며, 현재 아키텍처(Streamlit + Python + Clean Architecture)와 매우 잘 맞습니다.

## 2. 작동 원리 (Technical Approach)

### A. Context Manager (맥락 인식 엔진)

챗봇이 현재 화면을 이해하려면 **"현재 사용자가 무엇을 보고 있는가?"**를 실시간으로 파악해야 합니다. 이를 위해 `ContextManager`를 구현합니다.

*   **작동 방식**:
    1.  `st.session_state` 감지: 현재 선택된 탭이 무엇인지 확인 (예: `tab_idx`, `selected_tab`).
    2.  **데이터 추출**:
        *   **단일 종목 탭**: 현재 선택된 종목 코드, 가격, 기술적 지표, AI 리포트 내용 추출.
        *   **AI 스크리너 탭**: 스크리닝 결과 리스트(Top 5 종목, 추천 이유 등) 추출.
        *   **포트폴리오 탭**: 현재 포트폴리오 비중, 성과 데이터 추출.
    3.  **프롬프트 주입**: 사용자의 질문 앞에 이 "맥락 데이터"를 시스템 프롬프트(System Instruction)로 몰래 주입합니다.

### B. Gemini Chat Session (대화 기억)

단발성 질문/답변이 아니라, 이전 대화를 기억하는 **대화형 인터페이스**를 구현합니다.

*   **Gemini API 기능 활용**: `model.start_chat(history=...)` 사용하여 대화 맥락 유지.
*   **사이드바 UI**: `st.sidebar`에 채팅창을 배치하여 어느 탭에 있든 항상 대화 가능.

## 3. 사용자 경험 (UX Scenario)

### 상황 1: AI 스크리너 탭에서
*   **화면**: AI가 추천한 3개 종목이 떠 있음.
*   **사용자**: (사이드바에) "1위 종목이 삼성전자인데, 지금 진입해도 돼?"
*   **AI**: (스크리너 결과와 삼성전자 현재가 데이터를 확인 후) "네, 삼성전자는 현재 RSI 32로 과매도 구간이며 기관 수급이 들어오고 있어 진입 시점으로 매력적입니다..."

### 상황 2: 단일 종목 분석 탭에서
*   **화면**: SK하이닉스 차트와 AI 리포트가 떠 있음.
*   **사용자**: "이 종목의 가장 큰 리스크가 뭐야?"
*   **AI**: (현재 SK하이닉스를 보고 있음을 인지) "현재 SK하이닉스의 리스크 요인은..."

## 4. 구현 로드맵 (Phase D 제안)

이 기능을 **Phase D: The Mouth & Ears (AI Chatbot)**로 명명하고 진행할 것을 제안합니다.

### Step 1: Gemini Client 확장
- `chat` 세션 관리 기능 추가 (`start_chat`, `send_message`)

### Step 2: Context Manager 구현
- 각 탭별 데이터 추출 로직 구현 (`get_current_context()`)

### Step 3: UI 구현
- `app.py` 사이드바에 채널 UI 통합 (`st.chat_input`, `st.chat_message`)

---

## 5. 결론

사용자의 아이디어는 **매우 훌륭하며**, 기존 시스템의 활용도를 극대화할 수 있는 기능입니다. **즉시 구현을 시작하는 것을 추천합니다.**
