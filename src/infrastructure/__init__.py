"""
Infrastructure Layer
Clean Architecture: External Dependencies and Implementations
"""

# Market Data Gateways (Phase F-1)
from src.infrastructure.market_data import (
    YahooFinanceGateway,
    PyKRXGateway,
    FallbackStockDataGateway,
    GatewayFactory
)

# Sentiment Analyzers (Phase F-3)
from src.infrastructure.sentiment import (
    LLMSentimentAnalyzer,
    VaderSentimentAnalyzer,
    SentimentResult
)

# Repositories
from src.infrastructure.repositories.market_data_cache_repository import SQLiteMarketDataCache
from src.infrastructure.repositories.chat_history_repository import (
    SQLiteChatHistoryRepository,
    ChatHistoryEntry
)

__all__ = [
    # Market Data
    "YahooFinanceGateway",
    "PyKRXGateway",
    "FallbackStockDataGateway",
    "GatewayFactory",
    
    # Sentiment
    "LLMSentimentAnalyzer",
    "VaderSentimentAnalyzer",
    "SentimentResult",
    
    # Repositories
    "SQLiteMarketDataCache",
    "SQLiteChatHistoryRepository",
    "ChatHistoryEntry"
]

