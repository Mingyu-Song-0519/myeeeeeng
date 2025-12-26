"""
PyKRX Gateway
Clean Architecture: Infrastructure Layer

pykrx를 통한 한국 주식 데이터 수집 구현체
"""
import logging
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd

try:
    from pykrx import stock as pykrx_stock
    PYKRX_AVAILABLE = True
except ImportError:
    PYKRX_AVAILABLE = False

from src.domain.market_data.interfaces import IStockDataGateway

logger = logging.getLogger(__name__)


class PyKRXGateway(IStockDataGateway):
    """
    pykrx 데이터 게이트웨이 (한국 주식 전용)
    
    - KOSPI, KOSDAQ 모두 지원
    - 한국투자데이터(KRX)에서 직접 조회
    """
    
    @property
    def name(self) -> str:
        return "pykrx"
    
    def fetch_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """pykrx에서 OHLCV 데이터 조회"""
        if not PYKRX_AVAILABLE:
            logger.error("[PyKRXGateway] pykrx not installed")
            return None
        
        try:
            # 한국 주식 티커만 지원
            clean_ticker = self._clean_ticker(ticker)
            if not clean_ticker:
                logger.warning(f"[PyKRXGateway] Invalid KR ticker: {ticker}")
                return None
            
            # 날짜 계산
            if start and end:
                start_date = start.replace('-', '')
                end_date = end.replace('-', '')
            else:
                end_dt = datetime.now()
                period_days = self._parse_period(period)
                start_dt = end_dt - timedelta(days=period_days)
                start_date = start_dt.strftime('%Y%m%d')
                end_date = end_dt.strftime('%Y%m%d')
            
            # pykrx 조회
            df = pykrx_stock.get_market_ohlcv_by_date(
                start_date, 
                end_date, 
                clean_ticker
            )
            
            if df is None or df.empty:
                logger.warning(f"[PyKRXGateway] No data for {ticker}")
                return None
            
            # 컬럼명 정규화
            df = df.rename(columns={
                '시가': 'open',
                '고가': 'high',
                '저가': 'low',
                '종가': 'close',
                '거래량': 'volume'
            })
            
            logger.info(f"[PyKRXGateway] Fetched {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"[PyKRXGateway] Failed to fetch {ticker}: {e}")
            return None
    
    def is_available(self) -> bool:
        """pykrx 사용 가능 여부"""
        return PYKRX_AVAILABLE
    
    def supports_ticker(self, ticker: str) -> bool:
        """한국 주식 티커만 지원"""
        clean = self._clean_ticker(ticker)
        return clean is not None and len(clean) == 6 and clean.isdigit()
    
    def _clean_ticker(self, ticker: str) -> Optional[str]:
        """
        티커 정리 (한국 주식용)
        
        - '005930.KS' → '005930'
        - '035720.KQ' → '035720'
        - '005930' → '005930'
        """
        if '.KS' in ticker or '.KQ' in ticker:
            return ticker.split('.')[0]
        
        # 6자리 숫자 확인
        if ticker.isdigit() and len(ticker) == 6:
            return ticker
        
        return None
    
    def _parse_period(self, period: str) -> int:
        """기간 문자열을 일수로 변환"""
        period_map = {
            '1d': 1,
            '5d': 5,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825,
            'max': 3650
        }
        return period_map.get(period, 365)
