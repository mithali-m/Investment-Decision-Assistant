# Overview 

Investment Decision Assistant is a comprehensive GenAI and LLM-powered financial analytics platform built entirely in Python. It is designed to assist a wide range of users — from seasoned investors and financial analysts to students and busy professionals — in making informed, data-driven market decisions without requiring deep technical expertise. 

The platform integrates three core analytical dimensions: real-time stock price trend analysis driven by LSTM neural networks, financial statement analysis using Neo4j graph modeling, and news sentiment analysis powered by OpenAI's GPT-4o-mini. These three streams converge into a unified Streamlit web application that surfaces clear, actionable investment insights including buy, hold, and sell signals for target companies such as Tesla (TSLA), Nvidia (NVDA), and Amazon (AMZN). 

The system is built around an automated data pipeline that handles both an initial bulk load of historical data and ongoing incremental updates, ensuring the analysis always reflects the most current market state. A scheduler orchestrates these pipeline stages so the platform stays fresh without manual intervention.

# Features 
## 1. Stock Price Trend Analysis (LSTM) 
StockAnalysis.py implements a Long Short-Term Memory (LSTM) recurrent neural network that models the temporal dynamics of stock price behavior. LSTM architectures are specifically designed to capture long-range dependencies in sequential data, making them well-suited for financial time series where patterns separated by weeks or months can still carry predictive relevance. 

The model is trained on historical OHLCV (Open, High, Low, Close, Volume) data fetched via FetchData.py. It generates multi-step price forecasts and confidence estimates that are visualized as annotated overlays on the price chart within the Streamlit dashboard. 

• Trained on historical daily OHLCV price data per ticker <br>
• Produces short- and medium-term price trajectory forecasts <br>
• Outputs confidence estimates alongside point predictions <br>
• Automatically retrained as new incremental data is loaded

## 2. Financial Statement Analysis 
FinancialStatementAnalysis.py leverages Neo4j as its backbone for modeling the relationships between financial entities. Raw financial statements — balance sheets, income statements, and cash flow statements — are loaded into the graph database by InitialLoad.py and kept current by IncrementalLoad.py. 

The graph-native approach allows the system to traverse relationships that tabular databases cannot easily represent: how working capital changes propagate through liquidity ratios, or how multi-year revenue trends relate to industry peer performance. The analysis module queries these relationships to compute derived metrics and flag anomalies. 

• Ingests balance sheets, income statements, and cash flow statements <br>
• Stores financial entities and their relationships in Neo4j graph format <br>
• Computes key financial ratios: P/E, P/B, Debt-to-Equity, Free Cash Flow yield, and more <br>
• Enables graph traversal for peer comparison and cross-metric anomaly detection <br>
• Supports incremental updates without reprocessing historical records

## 3. News Sentiment Analysis 
FinancialNewsAnalysis.py integrates OpenAI's GPT-4o-mini to process financial news articles collected by FetchData.py. Each article is analyzed for sentiment polarity (positive, negative, neutral), topic classification (earnings, regulatory, product launch, macroeconomic), and named entity mentions. 

Sentiment scores are aggregated across articles into a per-ticker Composite Sentiment Index (CSI), weighted by recency and source relevance. This CSI provides a real-time pulse on market narrative and is a critical input to the decision engine. 

• Processes financial news articles through GPT-4o-mini LLM <br>
• Classifies sentiment polarity and news topic per article <br>
• Aggregates scores into a Composite Sentiment Index (CSI) per ticker <br>
• Applies recency weighting — recent news has greater influence on the score <br>
• Identifies divergences between price action and news sentiment

## 4. Unified Investment Decision Engine 
InvestmentDecisionAssistant.py is the core orchestrator. It ingests the outputs of all three analytical engines and applies an ensemble weighting strategy to produce a final investment recommendation. The output includes a primary signal (BUY, HOLD, or SELL), a confidence score, and a human-readable rationale that cites the dominant contributing factors. 

• Fuses LSTM price forecast, fundamental ratio scores, and sentiment CSI <br>
• Outputs a BUY / HOLD / SELL signal with a confidence percentage <br>
• Generates a plain-language rationale citing the top signal contributors <br>
• Surfaces risk flags: earnings uncertainty, high debt exposure, sentiment fragility

## 5. Automated Data Pipeline 
The data infrastructure is split across three modules. FetchData.py abstracts all external API communication. InitialLoad.py performs a one-time bulk load of years of historical data into Neo4j. IncrementalLoad.py appends only the latest data on each run, ensuring efficiency. PipelineScheduler.py ties these together by scheduling IncrementalLoad to execute at a configured interval, keeping the entire knowledge base current. 

• FetchData.py — unified API client for stock prices, financials, and news <br>
• InitialLoad.py — bulk historical data ingestion into Neo4j on first run <br>
• IncrementalLoad.py — daily delta ingestion appending only new records <br>
• PipelineScheduler.py — automated scheduling of the incremental pipeline 

## 7. Streamlit Web Application 
StreamlitApp.py delivers all analysis through an interactive browser-based interface. The app is built with Streamlit, making it immediately accessible without any frontend engineering. Users can search by ticker, explore price forecasts, review financial metrics, inspect sentiment trends, and view the final investment recommendation — all within a single, responsive UI. 

• Ticker search with instant analysis trigger <br>
• Interactive price chart with LSTM forecast overlay and confidence bands <br>
• Financial ratio dashboard with trend indicators <br>
• Sentiment timeline heatmap per ticker <br>
• Final signal card: BUY / HOLD / SELL with confidence score and rationale

# Technology Stack 
<b>Python 3.x </b>- Primary and sole programming language. All ten modules are written in Python. <br>
<b>Neo4j (Graph DB) </b>- Stores financial entities and relationships. Powers the financial statement analysis and enables graph-native ratio computation and peer comparison. <br>
<b>OpenAI GPT-4o-mini </b>- Large language model used in FinancialNewsAnalysis.py for sentiment classification, topic extraction, and natural language summarization of news articles. <br>
<b>LSTM (Keras / TensorFlow) </b>- Recurrent neural network architecture used in StockAnalysis.py for modeling price sequences and generating multi-step price forecasts. <br>
<b>Streamlit </b>- Python-native web framework used in StreamlitApp.py to render the interactive dashboard, charts, and investment signal UI. <br>
<b>Alpha Vantage API </b>- Primary external data source for daily stock prices and company financial statements accessed via FetchData.py. <br>
<b>News API (or equivalent) </b>- Source of financial news articles ingested by FetchData.py and processed by FinancialNewsAnalysis.py for sentiment analysis.

# Data Pipeline Architecture 
The following describes the end-to-end data flow from ingestion to decision output: 

<b>FetchData.py </b>- retrieves raw data: daily OHLCV prices, quarterly financial statements, and news articles from external APIs. <br>
<b>InitialLoad.py </b>- (first run only) bulk-loads this data into Neo4j, creating nodes for companies, financial periods, and news events, with labeled relationships between them. <br>
<b>PipelineScheduler.py </b>- triggers IncrementalLoad.py on a daily schedule, which appends only new records to the existing graph — avoiding redundant reprocessing. <br>
<b>StockAnalysis.py </b>- reads the price node sequence from Neo4j, prepares LSTM input features, runs inference, and writes forecast results. <br>
<b>FinancialStatementAnalysis.py </b>- traverses Neo4j financial nodes to compute ratios and trend signals. <br>
<b>FinancialNewsAnalysis.py </b>- sends news article text to GPT-4o-mini, receives sentiment labels, and aggregates them into per-ticker CSI scores. <br>
<b>InvestmentDecisionAssistant.py </b>- receives all three signal sets, applies ensemble weighting, and emits a final recommendation with rationale. <br>
<b>StreamlitApp.py </b>- renders all outputs through the interactive web UI.

# Disclaimer 
The Investment Decision Assistant is an informational and research tool. All outputs — including buy/sell/hold signals, price forecasts, sentiment scores, and financial summaries — are generated by automated models and algorithms. Nothing produced by this platform constitutes financial advice. Investment decisions carry risk and you may lose money. Always consult a qualified financial professional before making any investment. The authors and contributors of this project accept no liability for financial losses resulting from use of this tool.


