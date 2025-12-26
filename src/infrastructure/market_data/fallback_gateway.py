"""
Fallback Stock Data Gateway
Clean Architecture: Infrastructure Layer

다중 데이터 소스 Fallback 패턴 구현
"""
import logging
from typing import Optional, List
import pandas as pd

from src.domain.market_data.interfaces import (
    IStockDataGateway, 
    DataUnavailableError
)

logger = logging.getLogger(__name__)


class FallbackStockDataGateway(IStockDataGateway):
    """
    Fallback 패턴 게이트웨이
    
    여러 데이터 소스를 순차적으로 시도하여
    첫 번째 성공한 결과를 반환합니다.
    
    Example:
        gateways = [PyKRXGateway(), YahooFinanceGateway()]
        fallback = FallbackStockDataGateway(gateways)
        df = fallback.fetch_ohlcv("005930")  # pykrx 실패 시 yahoo 시도
    """
    
    def __init__(self, gateways: List[IStockDataGateway]):
        """
        Args:
            gateways: 우선순위 순서대로 정렬된 게이트웨이 리스트
        """
        self.gateways = gateways
        self._last_successful_gateway: Optional[str] = None
    
    @property
    def name(self) -> str:
        return "fallback"
    
    def fetch_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y"
    ) -> Optional[pd.DataFrame]:
        """
        다중 소스에서 순차적으로 데이터 조회 시도
        
        Returns:
            첫 번째 성공한 소스의 DataFrame
            
        Raises:
            DataUnavailableError: 모든 소스가 실패한 경우
        """
        errors = []
        
        for gateway in self.gateways:
            # 사용 불가능한 게이트웨이 건너뛰기
            if not gateway.is_available():
                logger.debug(f"[Fallback] Skipping unavailable: {gateway.name}")
                continue
            
            # 해당 티커를 지원하지 않는 게이트웨이 건너뛰기
            if not gateway.supports_ticker(ticker):
                logger.debug(f"[Fallback] {gateway.name} doesn't support {ticker}")
                continue
            
            try:
                df = gateway.fetch_ohlcv(ticker, start, end, period)
                
                if df is not None and not df.empty:
                    self._last_successful_gateway = gateway.name
                    logger.info(f"[Fallback] Success with {gateway.name} for {ticker}")
                    return df
                    
            except Exception as e:
                error_msg = f"{gateway.name}: {str(e)[:50]}"
                errors.append(error_msg)
                logger.warning(f"[Fallback] {error_msg}")
                continue
        
        # 모든 게이트웨이 실패
        error_detail = ", ".join(errors) if errors else "No available gateways"
        raise DataUnavailableError(f"All gateways failed for {ticker}: {error_detail}")
    
    def is_available(self) -> bool:
        """하나 이상의 게이트웨이가 사용 가능하면 True"""
        return any(g.is_available() for g in self.gateways)
    
    def get_last_successful_gateway(self) -> Optional[str]:
        """마지막으로 성공한 게이트웨이 이름"""
        return self._last_successful_gateway
    
    def get_available_gateways(self) -> List[str]:
        """현재 사용 가능한 게이트웨이 목록"""
        return [g.name for g in self.gateways if g.is_available()]
