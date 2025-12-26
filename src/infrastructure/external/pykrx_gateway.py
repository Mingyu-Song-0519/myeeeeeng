"""
PyKRX Gateway
한국 주식 수급 데이터(외국인/기관) 수집
Clean Architecture: Infrastructure Layer
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PyKRXGateway:
    """
    pykrx를 이용한 한국 주식 데이터 수집
    
    주요 기능:
    - 투자자별 매매동향 (외국인/기관/개인)
    - 거래량 및 거래대금 추이
    """
    
    def __init__(self):
        """PyKRX Gateway 초기화"""
        self._initialized = False
        self._init_pykrx()
    
    def _init_pykrx(self):
        """pykrx 라이브러리 초기화"""
        try:
            # pykrx 임포트 확인
            import pykrx
            self._initialized = True
            logger.info("[PyKRXGateway] Initialized successfully")
        except ImportError:
            logger.warning("[PyKRXGateway] pykrx not installed. Run: pip install pykrx")
    
    def is_available(self) -> bool:
        """서비스 사용 가능 여부"""
        return self._initialized
    
    def get_investor_trading(
        self,
        ticker: str,
        days: int = 20
    ) -> Optional[pd.DataFrame]:
        """
        투자자별 매매동향 조회 (외국인/기관/개인)
        
        Args:
            ticker: 종목 코드 (예: "005930.KS" 또는 "005930")
            days: 조회 기간 (일)
            
        Returns:
            DataFrame with columns: 날짜, 외국인순매수, 기관순매수, 개인순매수
            또는 None (데이터 없음 또는 오류 시)
        """
        if not self._initialized:
            logger.warning("[PyKRXGateway] Not initialized")
            return None
        
        try:
            from pykrx import stock
            
            # 티커에서 .KS, .KQ 제거
            clean_ticker = ticker.replace(".KS", "").replace(".KQ", "")
            
            # 날짜 범위 계산
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 날짜 포맷 변환 (YYYYMMDD)
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            # 투자자별 순매수 데이터 조회
            df = stock.get_market_trading_value_by_date(
                start_str,
                end_str,
                clean_ticker
            )
            
            if df is None or df.empty:
                logger.debug(f"[PyKRXGateway] No data for {ticker}")
                return None
            
            # 컬럼명 정리 (한글 컬럼명이 있는 경우)
            # pykrx 반환 형식: 날짜(index), 기관합계, 기타법인, 개인, 외국인합계, 전체
            df_result = pd.DataFrame({
                '날짜': df.index,
                '외국인순매수': df.get('외국인합계', df.get('외국인', 0)),
                '기관순매수': df.get('기관합계', df.get('기관', 0)),
                '개인순매수': df.get('개인', 0)
            })
            
            logger.debug(f"[PyKRXGateway] Fetched {len(df_result)} days for {ticker}")
            return df_result
            
        except Exception as e:
            logger.error(f"[PyKRXGateway] Failed to get investor trading for {ticker}: {e}")
            return None
    
    def get_investor_summary(
        self,
        ticker: str,
        days: int = 20
    ) -> Optional[Dict[str, Any]]:
        """
        투자자별 매매동향 요약
        
        Args:
            ticker: 종목 코드
            days: 조회 기간
            
        Returns:
            dict: {
                'foreign_net': 외국인 순매수 합계,
                'institution_net': 기관 순매수 합계,
                'individual_net': 개인 순매수 합계,
                'trend': '외국인/기관 동반 매수' 등
            }
        """
        df = self.get_investor_trading(ticker, days)
        
        if df is None or df.empty:
            return None
        
        # 순매수 합계 계산
        foreign_net = df['외국인순매수'].sum()
        institution_net = df['기관순매수'].sum()
        individual_net = df['개인순매수'].sum()
        
        # 추세 판단
        trend = ""
        if foreign_net > 0 and institution_net > 0:
            trend = "외국인/기관 동반 매수세"
        elif foreign_net < 0 and institution_net < 0:
            trend = "외국인/기관 동반 매도세"
        elif foreign_net > 0:
            trend = "외국인 매수 우위"
        elif institution_net > 0:
            trend = "기관 매수 우위"
        else:
            trend = "개인 매수 우위"
        
        return {
            'foreign_net': foreign_net,
            'institution_net': institution_net,
            'individual_net': individual_net,
            'trend': trend,
            'days': days
        }
    
    def detect_buying_streak(
        self,
        ticker: str,
        days: int = 20,
        streak_days: int = 3
    ) -> Dict[str, bool]:
        """
        연속 매수 추세 감지
        
        Args:
            ticker: 종목 코드
            days: 전체 조회 기간
            streak_days: 연속 일수 기준 (기본 3일)
            
        Returns:
            dict: {
                'foreign_streak': 외국인 N일 연속 매수 여부,
                'institution_streak': 기관 N일 연속 매수 여부
            }
        """
        df = self.get_investor_trading(ticker, days)
        
        if df is None or df.empty:
            return {'foreign_streak': False, 'institution_streak': False}
        
        # 최근 N일 데이터
        recent = df.tail(streak_days)
        
        # 연속 매수 체크
        foreign_streak = (recent['외국인순매수'] > 0).all()
        institution_streak = (recent['기관순매수'] > 0).all()
        
        return {
            'foreign_streak': foreign_streak,
            'institution_streak': institution_streak
        }


class MockPyKRXGateway(PyKRXGateway):
    """
    테스트용 Mock PyKRX Gateway
    """
    
    def __init__(self):
        """Mock 초기화 (pykrx 없이도 작동)"""
        self._initialized = True
    
    def get_investor_trading(self, ticker: str, days: int = 20) -> Optional[pd.DataFrame]:
        """Mock 데이터 반환"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        return pd.DataFrame({
            '날짜': dates,
            '외국인순매수': [1000000 * i for i in range(days)],  # 증가 추세
            '기관순매수': [500000 * i for i in range(days)],
            '개인순매수': [-1500000 * i for i in range(days)]  # 매도 추세
        })
