"""
Signal Generator Service
가중치 기반 매매 신호 생성 서비스
Clean Architecture: Application Layer

AI 예측 + 감성 분석 + 거래량 + 기관 수급을 종합하여 매매 신호 생성
"""
import logging
from datetime import datetime
from typing import Optional, Any

from src.domain.signal import TradingSignal, MarketRegime, SignalType

logger = logging.getLogger(__name__)


class SignalGeneratorService:
    """
    매매 신호 생성기 (라씨매매신호 스타일)
    
    가중치:
    - AI 신뢰도: 35%
    - 감성 점수: 25%
    - 거래량: 20%
    - 기관 수급: 20%
    """
    
    # 조건별 가중치
    WEIGHTS = {
        'ai_confidence': 0.35,     # AI 신뢰도: 35%
        'sentiment': 0.25,         # 감성: 25%
        'volume_spike': 0.20,      # 거래량: 20%
        'institution_buying': 0.20 # 기관 수급: 20%
    }
    
    def __init__(
        self,
        report_service: Optional[Any] = None,
        sentiment_service: Optional[Any] = None,
        pykrx_gateway: Optional[Any] = None,
        market_buzz_service: Optional[Any] = None
    ):
        """
        Args:
            report_service: InvestmentReportService (Phase A)
            sentiment_service: SentimentAnalysisService (Phase 18)
            pykrx_gateway: PyKRXGateway (Phase B)
            market_buzz_service: MarketBuzzService (Phase 21)
        """
        self.report_service = report_service
        self.sentiment_service = sentiment_service
        self.pykrx_gateway = pykrx_gateway
        self.market_buzz_service = market_buzz_service
    
    def generate_signal(
        self,
        ticker: str,
        stock_name: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> TradingSignal:
        """
        종합 매매 신호 생성 (가중치 기반)
        
        Args:
            ticker: 종목 코드
            stock_name: 종목명
            user_id: 사용자 ID (AI 리포트 개인화용)
            
        Returns:
            TradingSignal 객체
        """
        if stock_name is None:
            stock_name = ticker
        
        # 1. 조건별 점수 계산 (0-100)
        ai_score = self._calculate_ai_score(ticker, user_id)
        sentiment_score = self._calculate_sentiment_score(ticker)
        volume_score = self._calculate_volume_score(ticker)
        inst_score = self._calculate_institution_score(ticker)
        
        logger.debug(f"[SignalGen] {ticker}: AI={ai_score}, Sentiment={sentiment_score}, Volume={volume_score}, Inst={inst_score}")
        
        # 2. 가중 평균
        composite_score = (
            ai_score * self.WEIGHTS['ai_confidence'] +
            sentiment_score * self.WEIGHTS['sentiment'] +
            volume_score * self.WEIGHTS['volume_spike'] +
            inst_score * self.WEIGHTS['institution_buying']
        )
        
        # 3. 시장 상황 보정 (Phase 21 Market Buzz 활용)
        market_regime = self._get_market_regime()
        if market_regime == MarketRegime.BEAR:
            composite_score *= 0.7  # 약세장에서는 신호 강도 하향
            logger.debug(f"[SignalGen] Bear market adjustment: {composite_score:.1f}")
        
        # 4. 신호 판정
        signal_type = self._determine_signal_type(composite_score)
        
        # 5. 발동 조건 문자열 생성
        triggers, flags = self._generate_triggers(
            ai_score, sentiment_score, volume_score, inst_score
        )
        
        return TradingSignal(
            ticker=ticker,
            stock_name=stock_name,
            signal_type=signal_type,
            confidence=composite_score,
            triggers=triggers,
            generated_at=datetime.now(),
            ai_prediction_confident=flags['ai'],
            sentiment_positive=flags['sentiment'],
            volume_spike_detected=flags['volume'],
            institution_buying=flags['institution'],
            market_regime=market_regime,
            ai_score=ai_score,
            sentiment_score=sentiment_score,
            volume_score=volume_score,
            institution_score=inst_score
        )
    
    def _calculate_ai_score(self, ticker: str, user_id: Optional[str]) -> float:
        """AI 신뢰도 점수 (0-100)"""
        if not self.report_service:
            return 50.0  # 기본값
        
        try:
            report = self.report_service.generate_report(ticker, user_id=user_id)
            
            # AI 리포트 신뢰도를 그대로 사용
            score = report.confidence_score
            
            # 신호 타입에 따라 점수 조정
            if report.signal == SignalType.STRONG_BUY:
                score = min(100, score * 1.1)
            elif report.signal == SignalType.STRONG_SELL:
                score = max(0, score * 0.5)
            elif report.signal == SignalType.HOLD:
                score *= 0.7
            
            return score
            
        except Exception as e:
            logger.debug(f"[SignalGen] AI score failed for {ticker}: {e}")
            return 50.0
    
    def _calculate_sentiment_score(self, ticker: str) -> float:
        """감성 점수 (0-100)"""
        if not self.sentiment_service:
            return 50.0
        
        try:
            features = self.sentiment_service.get_sentiment_features(
                ticker=ticker,
                lookback_days=7
            )
            
            # 감성 점수 (0-1) → (0-100) 변환
            raw_score = features.get('sentiment_score', 0.5)
            score = raw_score * 100
            
            return score
            
        except Exception as e:
            logger.debug(f"[SignalGen] Sentiment score failed for {ticker}: {e}")
            return 50.0
    
    def _calculate_volume_score(self, ticker: str) -> float:
        """거래량 점수 (0-100)"""
        if not self.market_buzz_service:
            return 50.0
        
        try:
            buzz_score_obj = self.market_buzz_service.calculate_buzz_score(ticker)
            if not buzz_score_obj:
                return 50.0
            
            # 거래량 비율 기반 점수
            volume_ratio = getattr(buzz_score_obj, 'volume_ratio', 1.0)
            
            # 거래량이 평균 대비 2배 이상이면 높은 점수
            if volume_ratio >= 2.0:
                score = 90
            elif volume_ratio >= 1.5:
                score = 75
            elif volume_ratio >= 1.2:
                score = 60
            elif volume_ratio >= 0.8:
                score = 50
            else:
                score = 30
            
            return score
            
        except Exception as e:
            logger.debug(f"[SignalGen] Volume score failed for {ticker}: {e}")
            return 50.0
    
    def _calculate_institution_score(self, ticker: str) -> float:
        """기관 수급 점수 (0-100)"""
        if not self.pykrx_gateway:
            return 50.0
        
        # 한국 주식만 지원
        if not (ticker.endswith('.KS') or ticker.endswith('.KQ')):
            return 50.0
        
        try:
            summary = self.pykrx_gateway.get_investor_summary(ticker, days=20)
            if not summary:
                return 50.0
            
            # 외국인 + 기관 순매수 합계
            foreign_net = summary['foreign_net']
            inst_net = summary['institution_net']
            total_net = foreign_net + inst_net
            
            # 점수 계산 (순매수 금액 기반)
            # 1억 이상 매수 -> 90점
            # 5천만 이상 -> 75점
            # 양수 -> 60점
            # 0 -> 50점
            # 음수 -> 30점 이하
            if total_net >= 100_000_000:  # 1억
                score = 90
            elif total_net >= 50_000_000:  # 5천만
                score = 75
            elif total_net > 0:
                score = 60
            elif total_net == 0:
                score = 50
            else:
                score = max(20, 50 + (total_net / 100_000_000) * 30)  # 음수는 감점
            
            # 연속 매수 보너스
            streak = self.pykrx_gateway.detect_buying_streak(ticker, days=20, streak_days=3)
            if streak['foreign_streak'] and streak['institution_streak']:
                score = min(100, score * 1.15)
            
            return score
            
        except Exception as e:
            logger.debug(f"[SignalGen] Institution score failed for {ticker}: {e}")
            return 50.0
    
    def _get_market_regime(self) -> MarketRegime:
        """시장 상황 판단 (간이)"""
        # TODO: VIX, 시장 폭 등을 활용한 정교한 판단
        # 현재는 기본값
        return MarketRegime.NEUTRAL
    
    def _determine_signal_type(self, composite_score: float) -> SignalType:
        """종합 점수로 신호 타입 결정"""
        if composite_score >= 80:
            return SignalType.STRONG_BUY
        elif composite_score >= 65:
            return SignalType.BUY
        elif composite_score >= 40:
            return SignalType.HOLD
        elif composite_score >= 20:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL
    
    def _generate_triggers(
        self,
        ai_score: float,
        sentiment_score: float,
        volume_score: float,
        inst_score: float
    ) -> tuple:
        """발동 조건 문자열 및 플래그 생성"""
        triggers = []
        flags = {
            'ai': False,
            'sentiment': False,
            'volume': False,
            'institution': False
        }
        
        if ai_score >= 80:
            triggers.append(f"AI 신뢰도 {ai_score:.0f}점")
            flags['ai'] = True
        
        if sentiment_score >= 70:
            triggers.append(f"감성 긍정적 {sentiment_score:.0f}점")
            flags['sentiment'] = True
        
        if volume_score >= 70:
            triggers.append("거래량 급등")
            flags['volume'] = True
        
        if inst_score >= 70:
            triggers.append("기관 매수세")
            flags['institution'] = True
        
        return triggers, flags
