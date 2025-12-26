"""
Chat Domain Entities
Clean Architecture: Domain Layer
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class ChatMessage:
    """대화 메시지 (Value Object)"""
    role: str  # 'user' or 'model'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContextData:
    """
    현재 화면의 맥락 데이터 (DTO)
    
    챗봇이 "현재 무엇을 보고 있는지" 이해하기 위한 스냅샷 데이터
    """
    tab_name: str                  # 현재 탭 이름 (예: "단일 종목 분석", "AI 스크리너")
    market: str = "KR"             # 현재 시장 (KR/US)
    
    # 단일 종목 분석 탭용
    active_ticker: Optional[str] = None
    active_stock_name: Optional[str] = None
    current_price: Optional[float] = None
    technical_indicators: Optional[Dict[str, Any]] = None # RSI, MA 등
    ai_report_summary: Optional[str] = None
    
    # 스크리너 탭용
    screener_results: Optional[List[Dict[str, Any]]] = None # 추천 종목 리스트
    
    # 포트폴리오 탭용
    portfolio_summary: Optional[Dict[str, Any]] = None
    
    # Phase E: Action 실행용 참조 데이터
    available_tabs: List[str] = field(default_factory=list)  # 이동 가능한 탭 목록
    user_id: str = "default_user"  # Phase 20 프로필 연동용
    
    def to_prompt_string(self) -> str:
        """디버깅용 문자열 변환"""
        return f"Context(tab={self.tab_name}, ticker={self.active_ticker})"


class ChatSession:
    """대화 세션 엔티티"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
    
    def add_user_message(self, content: str):
        self.messages.append(ChatMessage(role='user', content=content))
    
    def add_model_message(self, content: str):
        self.messages.append(ChatMessage(role='model', content=content))
    
    def clear_history(self):
        self.messages = []
    
    @property
    def history_length(self) -> int:
        return len(self.messages)
