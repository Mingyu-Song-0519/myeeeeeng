"""
Market Data Service
Clean Architecture: Application Layer

시장 데이터 수집 및 캐싱 오케스트레이션 서비스
"""
import logging
from typing import Optional, List
from datetime import datetime, timedelta

from src.domain.market_data.interfaces import (
    OHLCV,
    IStockDataGateway,
    IMarketDataCache,
    DataUnavailableError,
    DataNotFoundError
)
from src.infrastructure.market_data.gateway_factory import GatewayFactory
from src.infrastructure.market_data.fallback_gateway import FallbackStockDataGateway

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    시장 데이터 수집 오케스트레이션 서비스
    
    기능:
    1. 다중 소스 Fallback 데이터 조회
    2. 캐싱 (TTL 기반)
    3. OHLCV 엔티티 변환
    """
    
    def __init__(
        self,
        gateways: Optional[List[IStockDataGateway]] = None,
        cache_repo: Optional[IMarketDataCache] = None,
        market: str = "KR"
    ):
        """
        Args:
            gateways: 데이터 게이트웨이 리스트 (None이면 자동 생성)
            cache_repo: 캐시 저장소 (None이면 캐싱 비활성화)
            market: 기본 시장 ("KR" 또는 "US")
        """
        if gateways:
            self.fallback_gateway = FallbackStockDataGateway(gateways)
        else:
            self.fallback_gateway = GatewayFactory.create_fallback_gateway(market)
        
        self.cache_repo = cache_repo
        self.market = market
        
        logger.info(f"[MarketDataService] Initialized for {market}, "
                   f"gateways: {self.fallback_gateway.get_available_gateways()}")
    
    def get_ohlcv(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1y",
        use_cache: bool = True
    ) -> OHLCV:
        """
        OHLCV 데이터 조회 (캐시 우선, 다중 소스 Fallback)
        
        Args:
            ticker: 종목 코드
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (YYYY-MM-DD)
            period: 조회 기간 (start/end 없을 때)
            use_cache: 캐시 사용 여부
            
        Returns:
            OHLCV 엔티티
            
        Raises:
            DataNotFoundError: 데이터가 없을 때
            DataUnavailableError: 모든 소스가 실패했을 때
        """
        # 날짜 기본값 설정
        if not end:
            end = datetime.now().strftime('%Y-%m-%d')
        if not start:
            period_days = self._parse_period(period)
            start_dt = datetime.now() - timedelta(days=period_days)
            start = start_dt.strftime('%Y-%m-%d')
        
        # 1. 캐시 확인
        if use_cache and self.cache_repo:
            cached = self.cache_repo.get(ticker, start, end)
            if cached and cached.is_valid():
                logger.debug(f"[MarketDataService] Cache hit for {ticker}")
                return cached
        
        # 2. 다중 소스 Fallback 조회
        try:
            df = self.fallback_gateway.fetch_ohlcv(ticker, start, end, period)
        except DataUnavailableError:
            raise
        
        if df is None or df.empty:
            raise DataNotFoundError(f"No data found for {ticker}")
        
        # 3. Domain 엔티티 변환
        source = self.fallback_gateway.get_last_successful_gateway() or "unknown"
        ohlcv = OHLCV.from_dataframe(ticker, df, source)
        
        # 4. 캐시 저장
        if use_cache and self.cache_repo and ohlcv.is_valid():
            self.cache_repo.save(ohlcv)
            logger.debug(f"[MarketDataService] Cached {ticker}")
        
        return ohlcv
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """최신 종가 조회"""
        try:
            ohlcv = self.get_ohlcv(ticker, period="5d")
            if ohlcv and len(ohlcv) > 0:
                return float(ohlcv.data['close'].iloc[-1])
        except Exception as e:
            logger.warning(f"[MarketDataService] Failed to get price for {ticker}: {e}")
        return None
    
    def get_multiple(self, tickers: List[str], period: str = "1y") -> dict:
        """
        여러 종목 OHLCV 일괄 조회
        
        Returns:
            {ticker: OHLCV} 딕셔너리 (실패한 종목은 제외)
        """
        results = {}
        for ticker in tickers:
            try:
                ohlcv = self.get_ohlcv(ticker, period=period)
                results[ticker] = ohlcv
            except Exception as e:
                logger.warning(f"[MarketDataService] Failed for {ticker}: {e}")
                continue
        
        return results
    
    def is_available(self) -> bool:
        """서비스 사용 가능 여부"""
        return self.fallback_gateway.is_available()
    
    def get_available_sources(self) -> List[str]:
        """사용 가능한 데이터 소스 목록"""
        return self.fallback_gateway.get_available_gateways()
    
    def invalidate_cache(self, ticker: str) -> bool:
        """특정 종목 캐시 무효화"""
        if self.cache_repo:
            return self.cache_repo.invalidate(ticker)
        return False
    
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
