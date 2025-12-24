"""
Level 3: Architecture Compliance Verification (아키텍처 준수 검증)
Clean Architecture 레이어 분리 및 DIP 준수 검증
"""
import sys
import ast
from pathlib import Path
from typing import List, Set, Dict
import traceback

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("🏛️ Level 3: Architecture Compliance Verification")
print("=" * 60)

results = {"passed": [], "failed": []}


def log_pass(msg: str):
    results["passed"].append(msg)
    print(f"✅ PASS: {msg}")


def log_fail(msg: str, error: str):
    results["failed"].append((msg, error))
    print(f"❌ FAIL: {msg}")
    print(f"   Error: {error}")


def get_imports_from_file(filepath: Path) -> Set[str]:
    """파일에서 import 구문 추출"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)

        return imports
    except Exception as e:
        print(f"Warning: {filepath} 파싱 실패 - {e}")
        return set()


# ============================================================
# 1. Layer 분리 검증
# ============================================================
print("\n" + "=" * 60)
print("📦 1. Clean Architecture Layer 분리")
print("=" * 60)

try:
    # Domain Layer 파일 확인
    domain_path = PROJECT_ROOT / "src" / "domain"
    if domain_path.exists():
        entity_files = list((domain_path / "entities").glob("*.py"))
        repo_interface_files = list((domain_path / "repositories").glob("*.py"))

        if len(entity_files) > 0:
            log_pass(f"Domain/Entities 존재 ({len(entity_files)}개 파일)")
        else:
            log_fail("Domain/Entities", "파일 없음")

        if len(repo_interface_files) > 0:
            log_pass(f"Domain/Repository Interfaces 존재 ({len(repo_interface_files)}개)")
        else:
            log_fail("Domain/Repository Interfaces", "파일 없음")
    else:
        log_fail("Domain Layer", "디렉토리 없음")

    # Infrastructure Layer 파일 확인
    infra_path = PROJECT_ROOT / "src" / "infrastructure"
    if infra_path.exists():
        repo_impl_files = list((infra_path / "repositories").glob("*.py"))

        if len(repo_impl_files) > 0:
            log_pass(f"Infrastructure/Repository 구현체 ({len(repo_impl_files)}개)")
        else:
            log_fail("Infrastructure/Repository", "파일 없음")
    else:
        log_fail("Infrastructure Layer", "디렉토리 없음")

    # Application Layer (Services) 파일 확인
    services_path = PROJECT_ROOT / "src" / "services"
    if services_path.exists():
        service_files = list(services_path.glob("*_service.py"))

        if len(service_files) > 0:
            log_pass(f"Application/Services 존재 ({len(service_files)}개)")
        else:
            log_fail("Application/Services", "파일 없음")
    else:
        log_fail("Application Layer (Services)", "디렉토리 없음")

except Exception as e:
    log_fail("Layer 분리 검증", str(e))
    traceback.print_exc()


# ============================================================
# 2. DIP (의존성 역전 원칙) 검증
# ============================================================
print("\n" + "=" * 60)
print("🔄 2. Dependency Inversion Principle (DIP)")
print("=" * 60)

try:
    # Domain Layer는 Infrastructure/Services에 의존하지 않아야 함
    domain_files = list((PROJECT_ROOT / "src" / "domain").rglob("*.py"))

    violation_count = 0
    for filepath in domain_files:
        imports = get_imports_from_file(filepath)

        # 금지된 import 체크
        for imp in imports:
            if imp.startswith("src.infrastructure"):
                log_fail(f"DIP 위반 ({filepath.name})", f"Infrastructure import: {imp}")
                violation_count += 1
            elif imp.startswith("src.services"):
                log_fail(f"DIP 위반 ({filepath.name})", f"Services import: {imp}")
                violation_count += 1

    if violation_count == 0:
        log_pass(f"Domain Layer DIP 준수 ({len(domain_files)}개 파일)")
    else:
        log_fail("Domain Layer DIP", f"{violation_count}개 위반")

except Exception as e:
    log_fail("DIP 검증", str(e))
    traceback.print_exc()


# ============================================================
# 3. Repository Pattern 준수
# ============================================================
print("\n" + "=" * 60)
print("📚 3. Repository Pattern 준수")
print("=" * 60)

try:
    from src.domain.repositories.interfaces import (
        IStockRepository,
        IPortfolioRepository,
        IKISRepository,
        INewsRepository,
        IIndicatorRepository
    )

    log_pass("Repository 인터페이스 정의 (5개)")

    # 구현체 확인
    from src.infrastructure.repositories.stock_repository import YFinanceStockRepository
    from src.infrastructure.repositories.portfolio_repository import (
        JSONPortfolioRepository,
        SessionPortfolioRepository
    )

    # 인터페이스 구현 확인
    if issubclass(YFinanceStockRepository, IStockRepository):
        log_pass("YFinanceStockRepository → IStockRepository 구현")
    else:
        log_fail("YFinanceStockRepository", "인터페이스 미구현")

    if issubclass(JSONPortfolioRepository, IPortfolioRepository):
        log_pass("JSONPortfolioRepository → IPortfolioRepository 구현")
    else:
        log_fail("JSONPortfolioRepository", "인터페이스 미구현")

    if issubclass(SessionPortfolioRepository, IPortfolioRepository):
        log_pass("SessionPortfolioRepository → IPortfolioRepository 구현")
    else:
        log_fail("SessionPortfolioRepository", "인터페이스 미구현")

except Exception as e:
    log_fail("Repository Pattern 검증", str(e))
    traceback.print_exc()


# ============================================================
# 4. Service Layer 의존성 주입 검증
# ============================================================
print("\n" + "=" * 60)
print("💉 4. Service Layer Dependency Injection")
print("=" * 60)

try:
    from src.services.portfolio_management_service import PortfolioManagementService
    from src.services.alert_orchestrator_service import AlertOrchestratorService

    # 생성자 시그니처 확인
    import inspect

    # PortfolioManagementService
    sig = inspect.signature(PortfolioManagementService.__init__)
    params = list(sig.parameters.keys())

    if "portfolio_repo" in params and "stock_repo" in params:
        log_pass("PortfolioManagementService DI (portfolio_repo, stock_repo)")
    else:
        log_fail("PortfolioManagementService DI", f"파라미터: {params}")

    # AlertOrchestratorService
    sig = inspect.signature(AlertOrchestratorService.__init__)
    params = list(sig.parameters.keys())

    if "stock_repo" in params:
        log_pass("AlertOrchestratorService DI (stock_repo)")
    else:
        log_fail("AlertOrchestratorService DI", f"파라미터: {params}")

except Exception as e:
    log_fail("Service DI 검증", str(e))
    traceback.print_exc()


# ============================================================
# 5. Entity 비즈니스 로직 검증
# ============================================================
print("\n" + "=" * 60)
print("🏢 5. Entity 비즈니스 로직 (Rich Domain Model)")
print("=" * 60)

try:
    from src.domain.entities.stock import StockEntity

    # StockEntity에 비즈니스 로직 메서드가 있는지 확인
    methods = [m for m in dir(StockEntity) if not m.startswith('_')]

    business_methods = [
        "get_price_range",
        "calculate_return",
        "calculate_volatility",
        "is_trending_up",
        "get_max_drawdown"
    ]

    found_methods = [m for m in business_methods if m in methods]

    if len(found_methods) >= 4:
        log_pass(f"StockEntity 비즈니스 로직 ({len(found_methods)}개 메서드)")
    else:
        log_fail("StockEntity 비즈니스 로직", f"메서드 부족: {found_methods}")

except Exception as e:
    log_fail("Entity 비즈니스 로직 검증", str(e))
    traceback.print_exc()


# ============================================================
# 6. Strangler Fig Pattern 검증 (Legacy ↔ Clean 공존)
# ============================================================
print("\n" + "=" * 60)
print("🌿 6. Strangler Fig Pattern (Legacy + Clean 공존)")
print("=" * 60)

try:
    # Legacy Analyzer (src/analyzers)
    legacy_path = PROJECT_ROOT / "src" / "analyzers"
    legacy_files = list(legacy_path.glob("*.py"))

    if len(legacy_files) > 0:
        log_pass(f"Legacy Analyzers 존재 ({len(legacy_files)}개)")
    else:
        log_fail("Legacy Analyzers", "파일 없음")

    # Clean Services (src/services)
    services_path = PROJECT_ROOT / "src" / "services"
    service_files = list(services_path.glob("*_service.py"))

    if len(service_files) > 0:
        log_pass(f"Clean Services 존재 ({len(service_files)}개)")
    else:
        log_fail("Clean Services", "파일 없음")

    # 공존 가능 여부 검증 (importable)
    try:
        from src.analyzers.volatility_analyzer import VolatilityAnalyzer
        from src.services.portfolio_management_service import PortfolioManagementService

        log_pass("Legacy + Clean 모듈 동시 Import 가능 (Strangler Fig)")
    except Exception as e:
        log_fail("Strangler Fig", f"Import 실패: {e}")

except Exception as e:
    log_fail("Strangler Fig Pattern 검증", str(e))
    traceback.print_exc()


# ============================================================
# 7. Phase 10-13 Clean Architecture 준수
# ============================================================
print("\n" + "=" * 60)
print("✨ 7. Phase 10-13 Clean Architecture 준수")
print("=" * 60)

try:
    # Phase 10: Domain Layer 확장
    from src.domain.entities.stock import StockEntity, PortfolioEntity, SignalEntity
    from src.domain.repositories.interfaces import IStockRepository, IPortfolioRepository

    log_pass("Phase 10: Domain Entities + Repository Interfaces")

    # Phase 11: Factor Analysis Service
    from src.analyzers.factor_analyzer import FactorAnalyzer, FactorScreener

    # FactorScreener가 DI를 사용하는지 확인
    import inspect
    sig = inspect.signature(FactorScreener.__init__)
    if "stock_repo" in sig.parameters:
        log_pass("Phase 11: FactorScreener DI 적용 (stock_repo)")
    else:
        log_fail("Phase 11: FactorScreener", "DI 미적용")

    # Phase 12: Social Trend Analyzer
    from src.analyzers.social_analyzer import SocialTrendAnalyzer
    log_pass("Phase 12: SocialTrendAnalyzer (Clean)")

    # Phase 13: Control Center (Dashboard Integration)
    control_center_path = PROJECT_ROOT / "src" / "dashboard" / "control_center.py"
    if control_center_path.exists():
        log_pass("Phase 13: Control Center Dashboard 통합")
    else:
        log_fail("Phase 13: Control Center", "파일 없음")

except Exception as e:
    log_fail("Phase 10-13 검증", str(e))
    traceback.print_exc()


# ============================================================
# 8. 순환 의존성 검증
# ============================================================
print("\n" + "=" * 60)
print("🔁 8. 순환 의존성 검증")
print("=" * 60)

try:
    # 간단한 순환 의존성 체크
    # Domain → Infrastructure (허용되지 않음)
    # Infrastructure → Domain (허용)
    # Services → Domain (허용)
    # Services → Infrastructure (허용)

    # Domain Layer에서 Infrastructure import 확인
    domain_files = list((PROJECT_ROOT / "src" / "domain").rglob("*.py"))
    circular_violations = []

    for filepath in domain_files:
        imports = get_imports_from_file(filepath)
        for imp in imports:
            if imp.startswith("src.infrastructure") or imp.startswith("src.services"):
                circular_violations.append(f"{filepath.name} → {imp}")

    if len(circular_violations) == 0:
        log_pass("순환 의존성 없음 (Domain Layer 독립)")
    else:
        for violation in circular_violations:
            log_fail("순환 의존성", violation)

except Exception as e:
    log_fail("순환 의존성 검증", str(e))
    traceback.print_exc()


# ============================================================
# 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("📋 Architecture Compliance 결과")
print("=" * 60)

total = len(results["passed"]) + len(results["failed"])
pass_rate = len(results["passed"]) / total * 100 if total > 0 else 0

print(f"\n✅ 통과: {len(results['passed'])}개")
print(f"❌ 실패: {len(results['failed'])}개")
print(f"📊 통과율: {pass_rate:.1f}%")

if results["failed"]:
    print("\n❌ 실패한 테스트:")
    for test, error in results["failed"]:
        print(f"  - {test}")
        print(f"    {error[:100]}")

print("\n" + "=" * 60)
if len(results["failed"]) == 0:
    print("🎉 Level 3 Architecture Compliance 완료!")
    print("   - Layer 분리 (Domain/Application/Infrastructure) ✅")
    print("   - DIP (Dependency Inversion Principle) ✅")
    print("   - Repository Pattern ✅")
    print("   - Service DI ✅")
    print("   - Rich Domain Model ✅")
    print("   - Strangler Fig Pattern ✅")
    print("   - 순환 의존성 없음 ✅")
else:
    print(f"⚠️ {len(results['failed'])}개 아키텍처 위반 발견.")
    print("   권장 조치:")
    print("   1. Domain Layer에서 Infrastructure import 제거")
    print("   2. Service Layer DI 적용")
    print("   3. Repository Pattern 구현")
print("=" * 60)

# Exit code
sys.exit(0 if len(results["failed"]) == 0 else 1)
