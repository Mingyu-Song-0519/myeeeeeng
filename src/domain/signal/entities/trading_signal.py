"""
Trading Signal Domain Entities
매매 신호 관련 도메인 엔티티
Clean Architecture: Domain Layer
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum

# SignalType은 ai_report에서 재사용
from src.domain.ai_report import SignalType


class MarketRegime(Enum):
    """시장 상황"""
    BULL = "상승장"       # 강세장
    BEAR = "하락장"       # 약세장
    NEUTRAL = "횡보장"    # 중립


@dataclass
class TradingSignal:
    """
    매매 신호 엔티티
    
    AI 예측, 감성 분석, 거래량, 기관 수급 등을 종합하여
    매매 신호를 생성합니다.
    """
    ticker: str
    stock_name: str
    signal_type: SignalType
    confidence: float  # 0-100 종합 신뢰도
    triggers: List[str]  # 발동 조건들 (예: "AI 신뢰도 85점", "감성 긍정적")
    generated_at: datetime = field(default_factory=datetime.now)
    
    # 신호 발동 조건 (각 항목의 충족 여부)
    ai_prediction_confident: bool = False  # AI 예측 신뢰도 80%+
    sentiment_positive: bool = False       # 감성 점수 0.7+
    volume_spike_detected: bool = False    # 거래량 급등
    institution_buying: bool = False       # 기관 순매수
    
    # 시장 상황
    market_regime: Optional[MarketRegime] = None
    
    # 개별 점수 (가중치 계산용)
    ai_score: float = 0.0
    sentiment_score: float = 0.0
    volume_score: float = 0.0
    institution_score: float = 0.0
    
    @property
    def is_actionable(self) -> bool:
        """실행 가능한 신호인지 (신뢰도 65% 이상)"""
        return self.confidence >= 65
    
    @property
    def trigger_count(self) -> int:
        """발동된 조건 개수"""
        return len(self.triggers)
    
    @property
    def signal_strength(self) -> str:
        """신호 강도"""
        if self.confidence >= 80:
            return "매우 강함"
        elif self.confidence >= 65:
            return "강함"
        elif self.confidence >= 50:
            return "보통"
        else:
            return "약함"
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            "ticker": self.ticker,
            "stock_name": self.stock_name,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "triggers": self.triggers,
            "generated_at": self.generated_at.isoformat(),
            "ai_prediction_confident": self.ai_prediction_confident,
            "sentiment_positive": self.sentiment_positive,
            "volume_spike_detected": self.volume_spike_detected,
            "institution_buying": self.institution_buying,
            "market_regime": self.market_regime.value if self.market_regime else None,
            "ai_score": self.ai_score,
            "sentiment_score": self.sentiment_score,
            "volume_score": self.volume_score,
            "institution_score": self.institution_score
        }
