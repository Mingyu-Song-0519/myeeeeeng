"""
Screener View
AI 종목 발굴 UI
Clean Architecture: Presentation Layer
"""
import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _get_screener_service():
    """ScreenerService 인스턴스 생성"""
    from src.services.screener_service import ScreenerService
    from src.services.signal_generator_service import SignalGeneratorService
    from src.infrastructure.external.pykrx_gateway import PyKRXGateway, MockPyKRXGateway
    
    # Signal Service
    signal_service = None
    try:
        from src.services.investment_report_service import InvestmentReportService
        from src.infrastructure.external.gemini_client import GeminiClient
        
        llm_client = GeminiClient()
        if not llm_client.is_available():
            from src.infrastructure.external.gemini_client import MockLLMClient
            llm_client = MockLLMClient()
        
        report_service = InvestmentReportService(llm_client=llm_client)
        signal_service = SignalGeneratorService(report_service=report_service)
    except Exception as e:
        logger.debug(f"Signal service init failed: {e}")
    
    # Profile Repo
    profile_repo = None
    try:
        from src.infrastructure.repositories.profile_repository import SQLiteProfileRepository
        profile_repo = SQLiteProfileRepository()
    except ImportError:
        pass
    
    # PyKRX Gateway
    pykrx_gateway = None
    try:
        gateway = PyKRXGateway()
        if gateway.is_available():
            pykrx_gateway = gateway
        else:
            pykrx_gateway = MockPyKRXGateway()
    except Exception as e:
        pykrx_gateway = MockPyKRXGateway()
    
    return ScreenerService(
        signal_service=signal_service,
        profile_repo=profile_repo,
        pykrx_gateway=pykrx_gateway
    )


def render_morning_picks():
    """오늘의 AI 추천주"""
    st.header("🌅 AI 모닝 픽")
    st.markdown("**AI가 발굴한 오늘의 추천 종목입니다.**")
    
    # 설정
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        market = st.selectbox(
            "시장 선택",
            ["KR", "US"],
            format_func=lambda x: "🇰🇷 한국" if x == "KR" else "🇺🇸 미국",
            key="screener_market"
        )
    
    with col2:
        top_n = st.number_input("추천 개수", min_value=3, max_value=10, value=5, key="screener_top_n")
    
    with col3:
        if st.button("🔍 종목 발굴", type="primary", use_container_width=True):
            st.session_state.screener_run = True
    
    # 필터 조건 안내
    with st.expander("📋 필터 조건", expanded=False):
        st.markdown("""
        **기술적 분석**
        - RSI < 40 (과매도 구간)
        
        **수급 분석** (한국 주식만)
        - 기관 3일 연속 매수
        
        **AI 종합 점수**
        - AI 예측 + 감성 분석 + 거래량 + 수급 종합
        """)
    
    # 스크리닝 실행
    if st.session_state.get('screener_run', False):
        st.session_state.screener_run = False
        
        # market 값은 위젯의 key로 자동 저장되므로 session_state에서 직접 가져옴
        market = st.session_state.get('screener_market', 'KR')
        
        with st.spinner(f"AI가 {market} 시장을 분석하는 중... (30초~1분 소요)"):
            try:
                service = _get_screener_service()
                user_id = st.session_state.get('user_id', 'default_user')
                
                picks = service.run_daily_screen(
                    user_id=user_id,
                    market=market,
                    top_n=top_n
                )
                
                st.session_state.screener_picks = picks
                # st.session_state.screener_market = market  ← 삭제! (위젯이 자동 관리)
                st.success(f"✅ {len(picks)}개 종목 발굴 완료!")
                
            except Exception as e:
                st.error(f"스크리닝 실패: {e}")
                logger.error(f"Screener failed: {e}")
                return
    
    # 결과 표시
    if 'screener_picks' in st.session_state:
        picks = st.session_state.screener_picks
        
        if not picks:
            st.info("조건을 만족하는 종목이 없습니다. 다른 시장을 선택하거나 나중에 다시 시도하세요.")
            return
        
        # 종목 리스트 표시 (return 삭제!)
        st.markdown("---")
        st.subheader(f"📊 추천 종목 ({len(picks)}개)")
        
        # 테이블 형식
        for i, pick in enumerate(picks, 1):
            with st.container():
                # 순위 배지
                rank_color = "#FFD700" if i == 1 else "#C0C0C0" if i == 2 else "#CD7F32" if i == 3 else "#E0E0E0"
                
                col_rank, col_info, col_score, col_detail = st.columns([0.5, 2, 1, 1])
                
                with col_rank:
                    st.markdown(f"""
                    <div style="
                        background-color: {rank_color};
                        border-radius: 50%;
                        width: 40px;
                        height: 40px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: bold;
                        font-size: 18px;
                    ">
                        {i}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_info:
                    st.markdown(f"### {pick.stock_name}")
                    st.caption(f"📌 {pick.ticker}")
                    
                    if pick.current_price:
                        change_color = "red" if pick.change_pct and pick.change_pct > 0 else "blue"
                        st.markdown(f"가격: **{pick.current_price:,.0f}원** "
                                  f"<span style='color:{change_color}'>({pick.change_pct:+.2f}%)</span>",
                                  unsafe_allow_html=True)
                
                with col_score:
                    st.metric("AI 점수", f"{pick.ai_score:.0f}")
                    st.caption(f"신뢰도: {pick.confidence:.0f}%")
                
                with col_detail:
                    st.text(pick.signal_type)
                    st.caption(pick.reason)
                
                # 세부 정보
                with st.expander(f"📈 {pick.stock_name} 상세 정보"):
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    
                    with detail_col1:
                        if pick.rsi:
                            st.metric("RSI", f"{pick.rsi:.1f}")
                    
                    with detail_col2:
                        if pick.pbr:
                            st.metric("PBR", f"{pick.pbr:.2f}")
                    
                    with detail_col3:
                        if pick.institution_streak:
                            st.success("✅ 기관 연속 매수")
                        else:
                            st.info("— 수급 정보 없음")
                
                st.markdown("---")
        
        # 내보내기 버튼
        if st.button("📥 CSV로 내보내기"):
            df = pd.DataFrame([
                {
                    '순위': i,
                    '종목명': p.stock_name,
                    '종목코드': p.ticker,
                    'AI점수': p.ai_score,
                    '신호': p.signal_type,
                    '현재가': p.current_price,
                    '등락률': p.change_pct,
                    'RSI': p.rsi,
                    'PBR': p.pbr,
                    '추천이유': p.reason
                }
                for i, p in enumerate(picks, 1)
            ])
            
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 다운로드",
                data=csv,
                file_name=f"ai_morning_picks_{st.session_state.get('screener_market', 'KR')}.csv",
                mime="text/csv"
            )
