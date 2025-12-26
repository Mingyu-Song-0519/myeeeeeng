"""
Chat Actions Domain Entities
Clean Architecture: Domain Layer
Phase E: AI Agentic Control
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


# 허용된 액션 목록 (보안: 화이트리스트)
ALLOWED_ACTIONS = frozenset({
    'switch_tab',
    'select_stock', 
    'run_screener',
    'run_analysis',
    'search_stock'
})


@dataclass(frozen=True)
class UIAction:
    """
    AI가 수행할 UI 작업 엔티티 (Value Object)
    
    Attributes:
        action_type: 액션 타입 (switch_tab, select_stock, run_screener, run_analysis, search_stock)
        params: 액션별 파라미터 딕셔너리
    """
    action_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['UIAction']:
        """
        딕셔너리에서 UIAction 객체 생성
        
        Args:
            data: {"action": "action_type", "params": {...}} 형식의 딕셔너리
            
        Returns:
            UIAction 객체 또는 None (파싱 실패 시)
        """
        if not isinstance(data, dict):
            logger.warning(f"[UIAction] Invalid data type: {type(data)}")
            return None
            
        action_type = data.get('action')
        if not action_type:
            logger.warning("[UIAction] Missing 'action' field")
            return None
        
        # 화이트리스트 검증
        if action_type not in ALLOWED_ACTIONS:
            logger.warning(f"[UIAction] Unknown action type: {action_type}")
            return None
            
        params = data.get('params', {})
        if not isinstance(params, dict):
            params = {}
            
        return cls(action_type=action_type, params=params)
    
    @classmethod
    def from_json(cls, json_str: str) -> Optional['UIAction']:
        """JSON 문자열에서 UIAction 객체 생성"""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.warning(f"[UIAction] JSON parsing failed: {e}")
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'action': self.action_type,
            'params': self.params
        }
    
    def is_valid(self) -> bool:
        """액션이 유효한지 검증"""
        return self.action_type in ALLOWED_ACTIONS


@dataclass
class ActionExecutionResult:
    """
    작업 실행 결과 DTO
    
    Attributes:
        success: 실행 성공 여부
        message: 사용자에게 표시할 메시지
        action: 실행된 UIAction
        redirect_needed: UI 리다이렉트 필요 여부 (탭 전환 등)
        data: 추가 결과 데이터 (종목 정보, 스크리너 결과 등)
    """
    success: bool
    message: str
    action: UIAction
    redirect_needed: bool = False
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'success': self.success,
            'message': self.message,
            'action': self.action.to_dict() if self.action else None,
            'redirect_needed': self.redirect_needed,
            'data': self.data
        }
    
    @classmethod
    def failure(cls, message: str, action: UIAction) -> 'ActionExecutionResult':
        """실패 결과 팩토리 메서드"""
        return cls(
            success=False,
            message=message,
            action=action,
            redirect_needed=False,
            data=None
        )
    
    @classmethod
    def success_with_redirect(
        cls, 
        message: str, 
        action: UIAction, 
        data: Optional[Dict[str, Any]] = None
    ) -> 'ActionExecutionResult':
        """리다이렉트가 필요한 성공 결과 팩토리 메서드"""
        return cls(
            success=True,
            message=message,
            action=action,
            redirect_needed=True,
            data=data
        )

