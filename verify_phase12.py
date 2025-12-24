"""
Phase 12 소셜 미디어 분석 검증 스크립트 (무료 API)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("🔥 Phase 12: 소셜 트렌드 분석 검증 (무료 API)")
print("="*60)

results = {"passed": [], "failed": []}


def log_pass(msg):
    results["passed"].append(msg)
    print(f"✅ PASS: {msg}")


def log_fail(msg, error):
    results["failed"].append((msg, error))
    print(f"❌ FAIL: {msg} - {error}")


# 1. 모듈 Import
print("\n" + "="*60)
print("📦 1. 모듈 Import 테스트")
print("="*60)

try:
    from src.analyzers.social_analyzer import (
        GoogleTrendsAnalyzer,
        SocialTrendAnalyzer,
        TrendCache,
        TrendData
    )
    log_pass("social_analyzer 모듈 import")
except Exception as e:
    log_fail("social_analyzer import", str(e))

# 2. pytrends 라이브러리 확인
print("\n" + "="*60)
print("🔧 2. pytrends 라이브러리 확인")
print("="*60)

try:
    import pytrends
    log_pass("pytrends 설치됨")
except ImportError:
    log_fail("pytrends", "pip install pytrends 필요")

# 3. Google Trends 조회 테스트
print("\n" + "="*60)
print("📈 3. Google Trends 조회 테스트")
print("="*60)

try:
    from src.analyzers.social_analyzer import GoogleTrendsAnalyzer
    
    analyzer = GoogleTrendsAnalyzer()
    
    if analyzer.available:
        log_pass("GoogleTrendsAnalyzer 초기화")
        
        # Tesla 트렌드 조회
        trend = analyzer.get_trend("Tesla", timeframe="today 1-m")
        
        if trend:
            log_pass(f"get_trend (Tesla: 관심도 {trend.current_interest})")
            
            print(f"\n  📊 Tesla 트렌드 분석:")
            print(f"     현재 관심도: {trend.current_interest}")
            print(f"     평균: {trend.avg_interest:.2f}")
            print(f"     최고점: {trend.peak_interest}")
            print(f"     추세: {trend.trend_direction}")
            print(f"     스파이크: {'🔥 감지!' if trend.spike_detected else '없음'}\n")
        else:
            log_fail("get_trend", "데이터 없음")
        
        # 비교 조회
        comp_df = analyzer.compare_trends(["Apple", "Microsoft", "Tesla"])
        if not comp_df.empty:
            log_pass(f"compare_trends ({len(comp_df)}일 데이터)")
        else:
            log_fail("compare_trends", "빈 데이터")
    else:
        log_fail("GoogleTrendsAnalyzer", "pytrends 없음")
        
except Exception as e:
    log_fail("Google Trends 조회", str(e))

# 4. 종목 관심도 분석
print("\n" + "="*60)
print("🚀 4. 종목 관심도 분석")
print("="*60)

try:
    from src.analyzers.social_analyzer import SocialTrendAnalyzer
    
    analyzer = SocialTrendAnalyzer()
    
    # TSLA 분석
    buzz = analyzer.analyze_stock_buzz("TSLA", "Tesla")
    
    if buzz:
        log_pass(f"analyze_stock_buzz (알림: {buzz['alert_level']})")
        
        print(f"\n  🔍 TSLA 소셜 버즈 분석:")
        print(f"     알림 수준: {buzz['alert_level']}")
        print(f"     설명: {buzz['description']}\n")
    else:
        log_fail("analyze_stock_buzz", "분석 실패")
        
except Exception as e:
    log_fail("종목 관심도 분석", str(e))

# 5. 밈주식 감지
print("\n" + "="*60)
print("🎯 5. 밈주식 감지 테스트")
print("="*60)

try:
    from src.analyzers.social_analyzer import SocialTrendAnalyzer
    
    analyzer = SocialTrendAnalyzer()
    
    # 유명 밈주식 후보
    watchlist = ["GME", "AMC", "TSLA"]
    meme_stocks = analyzer.detect_meme_stocks(watchlist, threshold=2.0)
    
    log_pass(f"detect_meme_stocks ({len(meme_stocks)}개 감지)")
    
    if meme_stocks:
        print(f"\n  🔥 밈주식 감지:")
        for stock in meme_stocks:
            print(f"     {stock['ticker']}: 관심도 {stock['interest']} (평균 {stock['avg']:.1f})")
        print()
        
except Exception as e:
    log_fail("밈주식 감지", str(e))

# 6. 캐싱 테스트
print("\n" + "="*60)
print("💾 6. 캐싱 시스템")
print("="*60)

try:
    from src.analyzers.social_analyzer import TrendCache, TrendData
    
    cache = TrendCache(ttl_minutes=60)
    
    # 테스트 데이터
    test_data = TrendData(
        keyword="TEST",
        current_interest=50,
        avg_interest=40.0,
        peak_interest=60,
        trend_direction="UP",
        spike_detected=False
    )
    
    cache.set("test_key", test_data)
    retrieved = cache.get("test_key")
    
    if retrieved and retrieved.keyword == "TEST":
        log_pass("캐싱 저장/조회")
    else:
        log_fail("캐싱", "조회 실패")
    
    cache.clear()
    log_pass("캐시 초기화")
    
except Exception as e:
    log_fail("캐싱 시스템", str(e))

# 결과 요약
print("\n" + "="*60)
print("📋 Phase 12 테스트 결과")
print("="*60)

total = len(results["passed"]) + len(results["failed"])
pass_rate = len(results["passed"]) / total * 100 if total > 0 else 0

print(f"\n✅ 통과: {len(results['passed'])}개")
print(f"❌ 실패: {len(results['failed'])}개")
print(f"📊 통과율: {pass_rate:.1f}%")

if results["failed"]:
    print("\n❌ 실패한 테스트:")
    for test, error in results["failed"]:
        print(f"  - {test}: {error[:50]}...")

print("\n" + "="*60)
if len(results["failed"]) == 0:
    print("🎉 Phase 12 소셜 트렌드 분석 검증 완료!")
    print("   - Google Trends API (무료) ✅")
    print("   - 종목 관심도 분석 ✅")
    print("   - 밈주식 감지 ✅")
    print("   - 캐싱 시스템 ✅")
else:
    print("⚠️ 일부 테스트 실패.")
    if "pytrends" in str(results["failed"]):
        print("\n💡 해결 방법: pip install pytrends")
print("="*60)
