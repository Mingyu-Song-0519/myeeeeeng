"""
Feature Engineering Service
Clean Architecture: Application Layer

기술적 지표 및 특성 생성 서비스
OHLCV 데이터에서 ML 모델 입력용 특성 벡터 생성
"""
import logging
from typing import Optional, List
import numpy as np
import pandas as pd

from src.domain.market_data.interfaces import OHLCV
from src.domain.prediction.value_objects import (
    TechnicalFeatures,
    MomentumFeatures,
    VolumeFeatures,
    FeatureVector
)

logger = logging.getLogger(__name__)


# 상수 정의
RSI_PERIOD_SHORT = 7
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
VOLATILITY_PERIOD = 20
MOMENTUM_PERIOD = 10
STOCH_PERIOD = 14


class FeatureEngineeringService:
    """
    기술적 지표 및 특성 생성 서비스
    
    OHLCV 데이터를 받아 다양한 기술적 지표를 계산하고
    ML 모델 입력용 특성 벡터로 변환합니다.
    """
    
    def create_technical_features(self, ohlcv: OHLCV) -> TechnicalFeatures:
        """
        OHLCV에서 기술적 특성 생성
        
        Args:
            ohlcv: OHLCV 데이터 엔티티
            
        Returns:
            TechnicalFeatures Value Object
        """
        df = ohlcv.to_dataframe()
        
        if len(df) < RSI_PERIOD:
            logger.warning(f"[FeatureEngineering] Insufficient data: {len(df)} rows")
            return TechnicalFeatures()
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # RSI
            rsi_14 = self._calculate_rsi(close, RSI_PERIOD)
            rsi_7 = self._calculate_rsi(close, RSI_PERIOD_SHORT) if len(close) >= RSI_PERIOD_SHORT else None
            
            # MACD
            macd_data = self._calculate_macd(close) if len(close) >= MACD_SLOW else {}
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(close) if len(close) >= BB_PERIOD else {}
            
            # Moving Averages
            sma_5 = self._calculate_sma(close, 5) if len(close) >= 5 else None
            sma_20 = self._calculate_sma(close, 20) if len(close) >= 20 else None
            sma_60 = self._calculate_sma(close, 60) if len(close) >= 60 else None
            sma_120 = self._calculate_sma(close, 120) if len(close) >= 120 else None
            
            ema_12 = self._calculate_ema(close, 12) if len(close) >= 12 else None
            ema_26 = self._calculate_ema(close, 26) if len(close) >= 26 else None
            
            # ATR
            atr_14 = self._calculate_atr(high, low, close) if len(close) >= ATR_PERIOD else None
            
            # Volatility
            volatility_20 = self._calculate_volatility(close) if len(close) >= VOLATILITY_PERIOD else None
            
            # Volume
            volume_ma_ratio = self._calculate_volume_ma_ratio(volume)
            
            # Momentum
            momentum_10 = self._calculate_momentum(close) if len(close) >= MOMENTUM_PERIOD else None
            roc_10 = self._calculate_roc(close) if len(close) >= MOMENTUM_PERIOD else None
            
            # Stochastic
            stoch = self._calculate_stochastic(high, low, close) if len(close) >= STOCH_PERIOD else {}
            
            return TechnicalFeatures(
                rsi_14=rsi_14,
                rsi_7=rsi_7,
                macd=macd_data.get('macd'),
                macd_signal=macd_data.get('signal'),
                macd_hist=macd_data.get('hist'),
                bb_upper=bb_data.get('upper'),
                bb_middle=bb_data.get('middle'),
                bb_lower=bb_data.get('lower'),
                bb_width=bb_data.get('width'),
                bb_pctb=bb_data.get('pctb'),
                sma_5=sma_5,
                sma_20=sma_20,
                sma_60=sma_60,
                sma_120=sma_120,
                ema_12=ema_12,
                ema_26=ema_26,
                atr_14=atr_14,
                volatility_20=volatility_20,
                volume_ma_ratio=volume_ma_ratio,
                momentum_10=momentum_10,
                roc_10=roc_10,
                stoch_k=stoch.get('k'),
                stoch_d=stoch.get('d')
            )
            
        except Exception as e:
            logger.error(f"[FeatureEngineering] Failed: {e}")
            return TechnicalFeatures()
    
    def create_momentum_features(self, ohlcv: OHLCV) -> MomentumFeatures:
        """모멘텀 특성 생성"""
        df = ohlcv.to_dataframe()
        close = df['close'].values
        
        if len(close) < 2:
            return MomentumFeatures()
        
        try:
            # 가격 변화율
            price_change_1d = (close[-1] / close[-2] - 1) * 100 if len(close) >= 2 else None
            price_change_5d = (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else None
            price_change_20d = (close[-1] / close[-21] - 1) * 100 if len(close) >= 21 else None
            
            # 52주 고저가
            if len(close) >= 252:
                high_52w = float(np.max(close[-252:]))
                low_52w = float(np.min(close[-252:]))
                current = close[-1]
                pct_from_high = (current / high_52w - 1) * 100
                pct_from_low = (current / low_52w - 1) * 100
            else:
                high_52w = low_52w = pct_from_high = pct_from_low = None
            
            return MomentumFeatures(
                price_change_1d=price_change_1d,
                price_change_5d=price_change_5d,
                price_change_20d=price_change_20d,
                high_52w=high_52w,
                low_52w=low_52w,
                pct_from_high=pct_from_high,
                pct_from_low=pct_from_low
            )
        except Exception as e:
            logger.error(f"[FeatureEngineering] Momentum failed: {e}")
            return MomentumFeatures()
    
    def create_volume_features(self, ohlcv: OHLCV) -> VolumeFeatures:
        """거래량 특성 생성"""
        df = ohlcv.to_dataframe()
        volume = df['volume'].values
        
        if len(volume) < 2:
            return VolumeFeatures()
        
        try:
            volume_change_1d = (volume[-1] / volume[-2] - 1) * 100 if volume[-2] > 0 else None
            volume_ma_5 = float(np.mean(volume[-5:])) if len(volume) >= 5 else None
            volume_ma_20 = float(np.mean(volume[-20:])) if len(volume) >= 20 else None
            
            # 거래량 급증 여부 (20일 평균 대비 2배 이상)
            volume_spike = None
            if volume_ma_20 and volume_ma_20 > 0:
                volume_spike = volume[-1] > volume_ma_20 * 2
            
            return VolumeFeatures(
                volume_change_1d=volume_change_1d,
                volume_ma_5=volume_ma_5,
                volume_ma_20=volume_ma_20,
                volume_spike=volume_spike
            )
        except Exception as e:
            logger.error(f"[FeatureEngineering] Volume failed: {e}")
            return VolumeFeatures()
    
    def create_feature_vector(self, ohlcv: OHLCV) -> FeatureVector:
        """통합 특성 벡터 생성"""
        from datetime import datetime
        
        technical = self.create_technical_features(ohlcv)
        momentum = self.create_momentum_features(ohlcv)
        volume = self.create_volume_features(ohlcv)
        
        return FeatureVector(
            ticker=ohlcv.ticker,
            date=datetime.now(),
            technical=technical,
            momentum=momentum,
            volume=volume
        )
    
    # === Private Methods: 지표 계산 ===
    
    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> Optional[float]:
        """RSI 계산"""
        if len(close) < period + 1:
            return None
        
        delta = np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(round(rsi, 2))
    
    def _calculate_macd(self, close: np.ndarray) -> dict:
        """MACD 계산"""
        if len(close) < MACD_SLOW:
            return {}
        
        ema_fast = self._ema(close, MACD_FAST)
        ema_slow = self._ema(close, MACD_SLOW)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, MACD_SIGNAL)
        histogram = macd_line - signal_line
        
        return {
            'macd': float(round(macd_line[-1], 4)),
            'signal': float(round(signal_line[-1], 4)),
            'hist': float(round(histogram[-1], 4))
        }
    
    def _calculate_bollinger_bands(self, close: np.ndarray) -> dict:
        """볼린저 밴드 계산"""
        if len(close) < BB_PERIOD:
            return {}
        
        sma = np.mean(close[-BB_PERIOD:])
        std = np.std(close[-BB_PERIOD:])
        
        upper = sma + (std * BB_STD)
        lower = sma - (std * BB_STD)
        current = close[-1]
        
        width = (upper - lower) / sma * 100
        pctb = (current - lower) / (upper - lower) if upper != lower else 0.5
        
        return {
            'upper': float(round(upper, 2)),
            'middle': float(round(sma, 2)),
            'lower': float(round(lower, 2)),
            'width': float(round(width, 2)),
            'pctb': float(round(pctb, 4))
        }
    
    def _calculate_sma(self, close: np.ndarray, period: int) -> Optional[float]:
        """단순 이동평균"""
        if len(close) < period:
            return None
        return float(round(np.mean(close[-period:]), 2))
    
    def _calculate_ema(self, close: np.ndarray, period: int) -> Optional[float]:
        """지수 이동평균 (최종값만)"""
        if len(close) < period:
            return None
        return float(round(self._ema(close, period)[-1], 2))
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """EMA 전체 시리즈"""
        multiplier = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[float]:
        """ATR 계산"""
        if len(close) < ATR_PERIOD + 1:
            return None
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        atr = np.mean(tr[-ATR_PERIOD:])
        return float(round(atr, 2))
    
    def _calculate_volatility(self, close: np.ndarray) -> Optional[float]:
        """변동성 (20일 수익률 표준편차)"""
        if len(close) < VOLATILITY_PERIOD + 1:
            return None
        
        returns = np.diff(close) / close[:-1]
        volatility = np.std(returns[-VOLATILITY_PERIOD:]) * np.sqrt(252)  # 연환산
        
        return float(round(volatility * 100, 2))
    
    def _calculate_volume_ma_ratio(self, volume: np.ndarray) -> Optional[float]:
        """거래량 / 20일 평균"""
        if len(volume) < 20:
            return None
        
        ma_20 = np.mean(volume[-20:])
        if ma_20 == 0:
            return None
        
        return float(round(volume[-1] / ma_20, 2))
    
    def _calculate_momentum(self, close: np.ndarray) -> Optional[float]:
        """모멘텀 (10일 전 대비 변화율)"""
        if len(close) < MOMENTUM_PERIOD + 1:
            return None
        
        momentum = (close[-1] / close[-MOMENTUM_PERIOD-1] - 1) * 100
        return float(round(momentum, 2))
    
    def _calculate_roc(self, close: np.ndarray) -> Optional[float]:
        """ROC (Rate of Change)"""
        return self._calculate_momentum(close)  # 동일 계산
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict:
        """스토캐스틱 계산"""
        if len(close) < STOCH_PERIOD:
            return {}
        
        highest = np.max(high[-STOCH_PERIOD:])
        lowest = np.min(low[-STOCH_PERIOD:])
        
        if highest == lowest:
            return {'k': 50.0, 'd': 50.0}
        
        k = ((close[-1] - lowest) / (highest - lowest)) * 100
        
        # %D는 %K의 3일 이평 (간이 계산)
        d = k  # 단순화
        
        return {
            'k': float(round(k, 2)),
            'd': float(round(d, 2))
        }
