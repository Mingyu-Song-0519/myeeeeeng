"""
Yahoo Finance Gateway
Clean Architecture: Infrastructure Layer

Yahoo Finance API를 통한 주식 데이터 수집 구현체
"""
import logging
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from src.domain.market_data.interfaces import IStockDataGateway

logger = logging.getLogger(__name__)


class YahooFinanceGateway(IStockDataGateway):
    """
    Yahoo Finance 데이터 게이트웨이
    
    - 한국 주식: 종목코드.KS (KOSPI), 종목코드.KQ (KOSDAQ)
    - 미국 주식: 티커 그대로 (AAPL, MSFT 등)
    """
    
    @property
    def name(self) -> str:
        return "yahoo_finance"
    
    def fetch_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """Yahoo Finance에서 OHLCV 데이터 조회"""
        if not YFINANCE_AVAILABLE:
            logger.error("[YahooGateway] yfinance not installed")
            return None
        
        try:
            # 한국 주식 티커 변환
            yf_ticker = self._convert_ticker(ticker)
            stock = yf.Ticker(yf_ticker)
            
            if start and end:
                df = stock.history(start=start, end=end)
            else:
                df = stock.history(period=period)
            
            if df is None or df.empty:
                logger.warning(f"[YahooGateway] No data for {ticker}")
                return None
            
            # 컬럼명 소문자로 정규화
            df.columns = [c.lower() for c in df.columns]
            
            logger.info(f"[YahooGateway] Fetched {len(df)} rows for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"[YahooGateway] Failed to fetch {ticker}: {e}")
            return None
    
    def is_available(self) -> bool:
        """yfinance 사용 가능 여부"""
        return YFINANCE_AVAILABLE
    
    def supports_ticker(self, ticker: str) -> bool:
        """Yahoo Finance는 대부분의 글로벌 주식 지원"""
        return True
    
    def _convert_ticker(self, ticker: str) -> str:
        """
        티커 변환 (한국 주식용)
        
        - '005930' → '005930.KS' (KOSPI)
        - '035720' → '035720.KQ' (KOSDAQ)
        - 'AAPL' → 'AAPL' (변경 없음)
        """
        # 이미 .KS 또는 .KQ가 붙어있으면 그대로
        if '.KS' in ticker or '.KQ' in ticker or '.' in ticker:
            return ticker
        
        # 6자리 숫자면 한국 주식으로 간주
        if ticker.isdigit() and len(ticker) == 6:
            # 기본적으로 KOSPI로 시도 (대부분의 대형주)
            return f"{ticker}.KS"
        
        return ticker
