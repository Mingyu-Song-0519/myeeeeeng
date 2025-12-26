"""
Prediction Domain Value Objects
Clean Architecture: Domain Layer

AI 예측 관련 Value Object 정의
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass(frozen=True)
class TechnicalFeatures:
    """
    기술적 지표 특성 (Value Object)
    
    모든 지표는 Optional - 데이터 부족 시 None
    """
    # 모멘텀 지표
    rsi_14: Optional[float] = None      # RSI (14일)
    rsi_7: Optional[float] = None       # RSI (7일, 단기)
    
    # MACD
    macd: Optional[float] = None        # MACD 라인
    macd_signal: Optional[float] = None # 시그널 라인
    macd_hist: Optional[float] = None   # MACD 히스토그램
    
    # 볼린저 밴드
    bb_upper: Optional[float] = None    # 상단 밴드
    bb_middle: Optional[float] = None   # 중간 밴드 (20일 이평)
    bb_lower: Optional[float] = None    # 하단 밴드
    bb_width: Optional[float] = None    # 밴드 폭
    bb_pctb: Optional[float] = None     # %B (현재가 위치)
    
    # 이동평균
    sma_5: Optional[float] = None       # 5일 이평
    sma_20: Optional[float] = None      # 20일 이평
    sma_60: Optional[float] = None      # 60일 이평
    sma_120: Optional[float] = None     # 120일 이평
    ema_12: Optional[float] = None      # 12일 지수이평
    ema_26: Optional[float] = None      # 26일 지수이평
    
    # 변동성
    atr_14: Optional[float] = None      # ATR (14일)
    volatility_20: Optional[float] = None  # 20일 변동성
    
    # 거래량
    volume_ma_ratio: Optional[float] = None  # 거래량/20일평균
    obv: Optional[float] = None         # OBV
    
    # 모멘텀
    momentum_10: Optional[float] = None  # 10일 모멘텀
    roc_10: Optional[float] = None       # 10일 ROC
    
    # 스토캐스틱
    stoch_k: Optional[float] = None     # %K
    stoch_d: Optional[float] = None     # %D
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환 (None 제외)"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def to_feature_vector(self) -> List[Optional[float]]:
        """ML 모델 입력용 벡터 변환"""
        return list(self.__dict__.values())
    
    def is_valid_for_prediction(self) -> bool:
        """예측에 필요한 최소 지표 존재 여부"""
        return self.rsi_14 is not None and self.sma_20 is not None


@dataclass(frozen=True)
class MomentumFeatures:
    """모멘텀 관련 특성"""
    price_change_1d: Optional[float] = None   # 1일 변화율
    price_change_5d: Optional[float] = None   # 5일 변화율
    price_change_20d: Optional[float] = None  # 20일 변화율
    
    high_52w: Optional[float] = None    # 52주 고가
    low_52w: Optional[float] = None     # 52주 저가
    pct_from_high: Optional[float] = None  # 고가 대비 %
    pct_from_low: Optional[float] = None   # 저가 대비 %


@dataclass(frozen=True)
class VolumeFeatures:
    """거래량 관련 특성"""
    volume_change_1d: Optional[float] = None   # 1일 거래량 변화
    volume_ma_5: Optional[float] = None        # 5일 평균 거래량
    volume_ma_20: Optional[float] = None       # 20일 평균 거래량
    volume_spike: Optional[bool] = None        # 거래량 급증 여부


@dataclass
class FeatureVector:
    """
    ML 모델 입력용 통합 특성 벡터 (Entity)
    """
    ticker: str
    date: datetime
    technical: TechnicalFeatures
    momentum: Optional[MomentumFeatures] = None
    volume: Optional[VolumeFeatures] = None
    
    # 레이블 (학습용)
    target_direction: Optional[int] = None  # 1: 상승, 0: 하락
    target_return: Optional[float] = None   # 실제 수익률
    
    def get_feature_names(self) -> List[str]:
        """특성 이름 목록"""
        return list(self.technical.__dict__.keys())
    
    def get_feature_values(self) -> List[Optional[float]]:
        """특성 값 목록"""
        return self.technical.to_feature_vector()
