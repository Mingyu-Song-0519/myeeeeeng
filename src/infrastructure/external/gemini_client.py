"""
LLM Client Infrastructure
Google Gemini API 클라이언트 및 인터페이스 정의
Clean Architecture: Infrastructure Layer
"""
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ILLMClient(ABC):
    """
    LLM 클라이언트 인터페이스 (DIP 준수)
    
    추후 Local LLM (Ollama, LLaMA) 전환 시 이 인터페이스만 구현하면 됨
    """
    
    @abstractmethod
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 사용자 프롬프트
            system_instruction: 시스템 지시 (선택)
            
        Returns:
            생성된 텍스트
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """서비스 사용 가능 여부 확인"""
        pass


class GeminiClient(ILLMClient):
    """
    Google Gemini API 클라이언트
    
    무료 티어 사용:
    - 분당 60회 요청 (RPM)
    - 일 1,500회 요청 (RPD)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Gemini API 키 (None이면 환경변수/Secrets에서 로드)
        """
        self.api_key = api_key
        self.model = None
        self._initialized = False
        
        self._init_client()
    
    def _init_client(self):
        """클라이언트 초기화"""
        try:
            import google.generativeai as genai
            
            # API 키 로드 순서: 인자 > Streamlit Secrets > 환경변수
            if self.api_key is None:
                self.api_key = self._load_api_key()
            
            if self.api_key is None:
                logger.warning("[GeminiClient] API key not found")
                return
            
            genai.configure(api_key=self.api_key)
            
            # gemini-2.0-flash 사용 시도 (사용자 환경에서 확인된 최신 안정 모델)
            try:
                # 사용 가능한 모델 목록에서 가장 적합한 모델 찾기
                available_models = [m.name.split('/')[-1] for m in genai.list_models() 
                                  if 'generateContent' in m.supported_generation_methods]
                
                if 'gemini-2.0-flash' in available_models:
                    model_name = 'gemini-2.0-flash'
                elif 'gemini-2.0-flash-lite' in available_models:
                    model_name = 'gemini-2.0-flash-lite'
                elif 'gemini-flash-latest' in available_models:
                    model_name = 'gemini-flash-latest'
                else:
                    # 목록에 없어도 시도해볼만한 기본값
                    model_name = 'gemini-2.0-flash'
                
                self.model = genai.GenerativeModel(model_name)
                logger.info(f"[GeminiClient] Selected model: {model_name}")
            except Exception as e:
                logger.warning(f"[GeminiClient] Model selection failed, using default: {e}")
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                
            self._initialized = True
            logger.info(f"[GeminiClient] Initialized successfully with model: {self.model.model_name}")
            
        except ImportError:
            logger.error("[GeminiClient] google-generativeai not installed")
        except Exception as e:
            logger.error(f"[GeminiClient] Init failed: {e}")
            raise # 에러를 상위로 전파하여 UI에서 보이게 함
    
    def _load_api_key(self) -> Optional[str]:
        """API 키 로드 (Streamlit Secrets 또는 환경변수)"""
        # 1. Streamlit Secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
                return st.secrets['GEMINI_API_KEY']
        except:
            pass
        
        # 2. 환경변수
        import os
        return os.environ.get('GEMINI_API_KEY')
    
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 사용자 프롬프트
            system_instruction: 시스템 지시 (선택)
            
        Returns:
            생성된 텍스트
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("GeminiClient not initialized. Check API key.")
        
        try:
            import google.generativeai as genai
            
            # 시스템 지시가 변경되었을 경우 모델 재설정 (native support 사용)
            if system_instruction:
                model_name = self.model.model_name.split('/')[-1]
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            response = model.generate_content(prompt)
            
            # 응답이 비어있거나 차단된 경우 처리
            if not response or not hasattr(response, 'text'):
                # candidate 피드백 확인
                if response.candidates and response.candidates[0].finish_reason:
                    reason = response.candidates[0].finish_reason
                    logger.warning(f"[GeminiClient] Blocked: {reason}")
                    return f"죄송합니다. 서비스 정책상 답변을 드릴 수 없습니다. (사유: {reason})"
                return "AI가 응답을 생성하지 못했습니다."
                
            return response.text
            
        except Exception as e:
            logger.error(f"[GeminiClient] Generation failed: {e}")
            raise

    
    def is_available(self) -> bool:
        """서비스 사용 가능 여부 확인"""
        return self._initialized and self.model is not None


class MockLLMClient(ILLMClient):
    """
    테스트용 Mock LLM 클라이언트
    
    개발/테스트 시 API 호출 없이 사용
    """
    
    def __init__(self, default_response: str = "Mock response"):
        self.default_response = default_response
    
    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Mock 응답 반환"""
        return f"""신호: BUY
신뢰도: 75
요약: 이 종목은 기술적으로 상승 추세에 있으며, 감성 분석 결과도 긍정적입니다.
논리: RSI가 과매도 구간을 벗어나 상승 중이며, 최근 뉴스 감성이 긍정적입니다. 거래량도 증가 추세입니다.
"""
    
    def is_available(self) -> bool:
        """항상 사용 가능"""
        return True
