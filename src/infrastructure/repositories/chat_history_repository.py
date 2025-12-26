"""
Chat History Repository
Clean Architecture: Infrastructure Layer

챗봇 대화 이력 및 분석 결과 저장소
"""
import logging
import sqlite3
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChatHistoryEntry:
    """대화 이력 항목"""
    user_id: str
    ticker: str
    stock_name: str
    signal_type: str  # BUY, SELL, HOLD
    confidence_score: float
    summary: str
    created_at: datetime
    
    def to_dict(self) -> dict:
        return {
            'user_id': self.user_id,
            'ticker': self.ticker,
            'stock_name': self.stock_name,
            'signal_type': self.signal_type,
            'confidence_score': self.confidence_score,
            'summary': self.summary,
            'created_at': self.created_at.isoformat()
        }


class IChatHistoryRepository:
    """
    대화 이력 저장소 인터페이스
    """
    
    def save_report(self, user_id: str, ticker: str, stock_name: str,
                   signal_type: str, confidence: float, summary: str) -> bool:
        """분석 리포트 저장"""
        raise NotImplementedError
    
    def get_recent_reports(self, user_id: str, limit: int = 5) -> List[ChatHistoryEntry]:
        """최근 분석 이력 조회"""
        raise NotImplementedError
    
    def get_reports_by_ticker(self, ticker: str, limit: int = 10) -> List[ChatHistoryEntry]:
        """특정 종목의 분석 이력"""
        raise NotImplementedError


class SQLiteChatHistoryRepository(IChatHistoryRepository):
    """
    SQLite 기반 대화 이력 저장소
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Args:
            db_path: 데이터베이스 경로 (None이면 기본 경로)
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "chat_history.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"[ChatHistoryRepo] Initialized at {self.db_path}")
    
    def save_report(
        self,
        user_id: str,
        ticker: str,
        stock_name: str,
        signal_type: str,
        confidence: float,
        summary: str
    ) -> bool:
        """분석 리포트 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_history
                    (user_id, ticker, stock_name, signal_type, confidence_score, summary, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    ticker,
                    stock_name,
                    signal_type,
                    confidence,
                    summary,
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            logger.debug(f"[ChatHistoryRepo] Saved report: {ticker} for {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"[ChatHistoryRepo] Save failed: {e}")
            return False
    
    def get_recent_reports(self, user_id: str, limit: int = 5) -> List[ChatHistoryEntry]:
        """최근 분석 이력 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, ticker, stock_name, signal_type, 
                           confidence_score, summary, created_at
                    FROM chat_history
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
                
                rows = cursor.fetchall()
                
            return [
                ChatHistoryEntry(
                    user_id=row[0],
                    ticker=row[1],
                    stock_name=row[2],
                    signal_type=row[3],
                    confidence_score=row[4],
                    summary=row[5],
                    created_at=datetime.fromisoformat(row[6])
                )
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"[ChatHistoryRepo] Get recent failed: {e}")
            return []
    
    def get_reports_by_ticker(self, ticker: str, limit: int = 10) -> List[ChatHistoryEntry]:
        """특정 종목의 분석 이력"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, ticker, stock_name, signal_type, 
                           confidence_score, summary, created_at
                    FROM chat_history
                    WHERE ticker = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (ticker, limit))
                
                rows = cursor.fetchall()
                
            return [
                ChatHistoryEntry(
                    user_id=row[0],
                    ticker=row[1],
                    stock_name=row[2],
                    signal_type=row[3],
                    confidence_score=row[4],
                    summary=row[5],
                    created_at=datetime.fromisoformat(row[6])
                )
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"[ChatHistoryRepo] Get by ticker failed: {e}")
            return []
    
    def get_stats(self, user_id: Optional[str] = None) -> dict:
        """통계 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if user_id:
                    cursor.execute("""
                        SELECT COUNT(*), COUNT(DISTINCT ticker)
                        FROM chat_history WHERE user_id = ?
                    """, (user_id,))
                else:
                    cursor.execute("""
                        SELECT COUNT(*), COUNT(DISTINCT ticker), COUNT(DISTINCT user_id)
                        FROM chat_history
                    """)
                
                row = cursor.fetchone()
            
            if user_id:
                return {
                    'total_reports': row[0],
                    'unique_tickers': row[1]
                }
            return {
                'total_reports': row[0],
                'unique_tickers': row[1],
                'unique_users': row[2]
            }
            
        except Exception:
            return {'total_reports': 0}
    
    def _init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    stock_name TEXT,
                    signal_type TEXT,
                    confidence_score REAL,
                    summary TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON chat_history(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON chat_history(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON chat_history(created_at)")
            conn.commit()
