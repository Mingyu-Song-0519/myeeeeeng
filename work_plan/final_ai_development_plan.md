# 🧠 Final AI Investment Assistant Development Plan

> **Goal**: Create an autonomous AI agent that analyzes stocks, recommends investments, and determines optimal buy/sell timing.
> **Philosophy**: Zero Cost Operation (Free Tier API) + Hybrid-Ready Architecture.

---

## 🏗️ Architecture Overview

The system evolves from a passive dashboard to an active **AI Agent**.

### Current vs. Target Architecture

| Component | Current Status | Target State (Final AI) |
| :--- | :--- | :--- |
| **Analysis Engine** | Simple Rules (RSI < 30) | **LLM (Gemini Pro)** + Technical + Fundamental |
| **Data Source** | Price History (yfinance) | Price + **Supply/Demand (Foregin/Inst)** + **Financials** |
| **Decision Logic** | Hardcoded Thresholds | **Context-Aware Reasoning** (Market Regime + Sector Trend) |
| **Execution** | Manual Watchlist | **AI Screener** (Auto-Discovery) -> **Paper Trader** (Simulation) |

---

## 🚀 Phase A: The Brain (AI Integration)
**Goal**: Integrate Google Gemini API to generate qualitative investment reports.

### A-1. LLM Service Infrastructure
- [ ] **Infrastructure**: Setup `GeminiClient` with API Key management (Streamlit Secrets).
- [ ] **Prompt Engineering**: Design prompts for "Investment Analyst Persona".
    - *Input*: Technical Indicators (RSI, MACD), Price Trend, Sector Info.
    - *Output*: Buy/Sell/Hold rating (1-5 star) + Reasoning summary.

### A-2. "Ask AI" Feature
- [ ] Add "🤖 AI Analysis" button to Stock Detail page.
- [ ] Generate real-time reports: *"Based on the RSI divergence and recent uptrend, this stock shows strong momentum..."*

---

## 🚀 Phase B: The Context (Data Enrichment & Signal Logic)
**Goal**: Feed the AI with "Alpha" data and implement robust buy/sell logic.

### B-1. Supply & Demand (Smart Money)
- [ ] **Data Source**: Integrate data for **Foreigner & Institution Net Buying**.
    - *Note*: Crucial for Korean market (KOSPI/KOSDAQ). Use `pykrx` or KIS API.
- [ ] **Visualization**: Add "Investor Trend" chart (Individual vs Foreigner vs Inst).

### B-2. Signal Logic & Confidence Intervals
- [ ] **Signal Generator**: Implement the "Buy" trigger logic.
    - *Condition*: `AI Prediction > 80% Confidence` AND `Sentiment Score > 0.7` AND `Volume Spike detected`.
    - *Reference*: ThinkPool's "RaSi" signals.
- [ ] **Confidence Scoring**: Display the AI's confidence level (e.g., "Strong Buy (92%)").

### B-3. Fundamental Health
- [ ] **Data Source**: Fetch PER, PBR, ROE, Operating Margin.
- [ ] **Evaluation**: Compare with sector averages (Undervalued/Overvalued check).

---

## 🚀 Phase C: The Hands (Execution & Personalization)
**Goal**: Automate discovery and tailor recommendations to the user.

### C-1. AI Screener (Morning Report)
- [ ] **Logic**: Scan 2000+ stocks for specific criteria everyday.
    - *Example*: `RSI < 35` AND `Inst Buy > 3 days` AND `PBR < 1.0`.
- [ ] **Auto-Report**: AI generates a "Morning Pick" list with top 3 recommendations.

### C-2. Personalization Engine
- [ ] **Profile Integration**: Filter recommendations based on user's investment profile.
    - *Aggressive*: High volatility + High Sentiment Score.
    - *Conservative*: Stable supply/demand + High Prediction Confidence.
- [ ] **Custom Ranking**: Re-rank the "Morning Pick" list based on user preferences.

### C-3. Paper Trading (Simulation)
- [ ] **Virtual Account**: Create a simulated portfolio tracking system.
- [ ] **Auto-Execution Logic**: If AI Score > 80, "Buy" virtual stock.
- [ ] **Performance Review**: Track AI's win rate and profit over 1 month.

---

## 🔮 Future Expansion: Local LLM Hybrid
**Goal**: Transition to high-performance Local LLM for privacy and depth.

### Hybrid Architecture Plan
1.  **Local PC (The Brain Server)**
    - Runs OLLAMA (Llama 3 / Mistral).
    - Performs deep analysis on 100+ stocks overnight.
    - Pushes results to GitHub (`analysis_results.json`) or Database.

2.  **Streamlit Cloud (The Face)**
    - Fetches pre-computed analysis from the shared storage.
    - Displays results to the user (Serverless & Free).

---

## 📅 Suggested Roadmap

| Step | Task | Duration | Impact |
| :--- | :--- | :--- | :--- |
| **1** | **Gemini API & LLM Service** | 2-3 Days | ⭐⭐⭐⭐⭐ (Immediate AI Intelligence) |
| **2** | **Fundamentals & Supply Data** | 3-4 Days | ⭐⭐⭐⭐ (Data Depth) |
| **3** | **AI Screener (Finding Stocks)** | 3-4 Days | ⭐⭐⭐⭐⭐ (Proactive Recommendation) |
| **4** | **Paper Trading System** | 4-5 Days | ⭐⭐⭐ (Verification) |

**Recommended Next Step**: Start with **Step 1 (Gemini API Integration)** to give your platform a "Voice".
