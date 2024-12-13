# Answers for all tickers provided in the question
# Main One
# Stock data, financial data and news for the ticker is extracted from neo4j for analysis
# Data is chunked and summerzied to follow chain-of-thought prompt to provide clear recommendation for investment
# Also provides buy/sell signals of the company stock data
# GPT 4o mini is used

import streamlit as st
import openai
from neo4j import GraphDatabase
import json
import pandas as pd
import matplotlib.pyplot as plt
import neo4j.time
from dotenv import load_dotenv
import os

load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j Driver
driver = GraphDatabase.driver(uri=NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Function to extract ticker from question
def extract_ticker_with_llm(question):
    messages = [
        {"role": "system", "content": "You are a financial assistant that extracts stock tickers from user questions."
                                      "Return only the stock tickers as a comma-separated list."},
        {"role": "user", "content": f"Extract the stock tickers from this question: {question}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    extracted_tickers = response.choices[0].message.content.strip()
    return [ticker.strip() for ticker in extracted_tickers.split(",") if ticker.strip()]


# Function to fetch data from Neo4j
def fetch_data(query, ticker):
    with driver.session() as session:
        result = session.run(query, {"ticker": ticker})
        data = []
        for record in result:
            record_dict = record.data()
            # Handle Neo4j's time.Date objects and convert to Python-native datetime.date
            for key in ["CurrentDate", "NextDate"]:
                if key in record_dict and isinstance(record_dict[key], neo4j.time.Date):
                    record_dict[key] = record_dict[key].to_native()  # Convert to datetime.date
            data.append(record_dict)
        return data


# Function to chunk the data into smaller parts
def chunk_data(data, max_chunk_size=16000):
    chunks = []
    current_chunk = []
    current_length = 0

    for record in data:
        record_str = json.dumps(record, default=str)
        record_length = len(record_str)

        if current_length + record_length > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
            current_length = 0

        current_chunk.append(record)
        current_length += record_length

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_all_data(stock_data, financial_data, news_data, max_chunk_size=16000):
    return {
        "stock_data_chunks": chunk_data(stock_data, max_chunk_size),
        "financial_data_chunks": chunk_data(financial_data, max_chunk_size),
        "news_data_chunks": chunk_data(news_data, max_chunk_size),
    }


def answer_user_question(question, stock_data, financial_data, news_data, max_chunk_size=16000):
    # Chunk the data
    chunked_data = chunk_all_data(stock_data, financial_data, news_data, max_chunk_size)

    all_responses = []

    # Process each chunk of stock data
    for i, chunk in enumerate(chunked_data["stock_data_chunks"]):
        chunk_str = json.dumps(chunk, default=str, indent=2)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial assistant that analyzes stock data."},
                {"role": "user",
                 "content": f"Analyze the following stock data chunk:\n{chunk_str}\n\nQuestion: {question}"}
            ]
        )
        all_responses.append(response.choices[0].message.content)

    # Process each chunk of financial data
    for i, chunk in enumerate(chunked_data["financial_data_chunks"]):
        chunk_str = json.dumps(chunk, default=str, indent=2)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial assistant that analyzes financial statements."},
                {"role": "user",
                 "content": f"Analyze the following financial data chunk:\n{chunk_str}\n\nQuestion: {question}"}
            ]
        )
        all_responses.append(response.choices[0].message.content)

    # Process each chunk of news data
    for i, chunk in enumerate(chunked_data["news_data_chunks"]):
        chunk_str = json.dumps(chunk, default=str, indent=2)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial assistant that analyzes news sentiment."},
                {"role": "user",
                 "content": f"Analyze the following news data chunk:\n{chunk_str}\n\nQuestion: {question}"}
            ]
        )
        all_responses.append(response.choices[0].message.content)

    # Combine all responses
    combined_answers = "\n".join(all_responses)

    # Use OpenAI GPT to analyze and re-evaluate the combined responses with a focus on sentiment
    chain_of_thought_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial assistant that re-evaluates combined data from stock analysis, financial statements, "
                    "and news sentiment analysis to provide step-by-step reasoning and a final recommendation."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here is the analysis from different datasets:\n{combined_answers}\n\n"
                    f"The user asked the following question:\n{question}\n\n"
                    "Please re-evaluate the analyses using chain-of-thought reasoning to provide a final response."
                    "Use the following steps:\n"
                    "1. Break down the insights provided by the analysis of stock data, financial data, and news sentiment.\n"
                    "2. Identify patterns, outliers, or trends in the stock data (open price, close price, and volume).\n"
                    "3. Analyze financial data, focusing on balance sheets, income statements, and cash flows.\n"
                    "4. Incorporate sentiment analysis from the news data (positive, negative, or neutral sentiment).\n"
                    "5. Synthesize all insights to answer the user's question with a clear recommendation."
                )
            }
        ]
    )

    return chain_of_thought_response.choices[0].message.content


def summary_analysis(data, question):
    summary_response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial assistant that re-evaluates combined data from stock analysis, financial statements, "
                    "and news sentiment analysis to provide step-by-step reasoning and a final recommendation."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here is the analysis from different datasets:\n{data}\n\n"
                    f"The user asked the following question:\n{question}\n\n"
                    "Please compare the analyses using chain-of-thought reasoning to provide a final response."
                    "Use the following steps:\n"
                    "1. Break down the insights provided by the analysis of stock data, financial data, and news sentiment.\n"
                    "2. Identify patterns, outliers, or trends in the stock data (open price, close price, and volume).\n"
                    "3. Analyze financial data, focusing on balance sheets, income statements, and cash flows.\n"
                    "4. Incorporate sentiment analysis from the news data (positive, negative, or neutral sentiment).\n"
                    "5. Synthesize all insights to answer the user's question with a clear recommendation."
                )
            }
        ]
    )
    return summary_response.choices[0].message.content


# Main function
def invest_decision_main(user_question):
    # Answer the user's question based on all the data
    try:
        # Fetch stock data from Neo4j
        tickers = extract_ticker_with_llm(user_question)
        print(f"Extracted tickers: {tickers}")

        if not tickers:
            print("No stock tickers were detected in your question. Please try again.")
            return

        all_answers = []
        for ticker in tickers:
            print(f"\nProcessing data for ticker: {ticker}")
            stock_data_query = '''
                        MATCH (s:Stock_Ticker {ticker: $ticker})-[:HAS_DAILY_PRICE]->(p:DailyPrice)
                        RETURN 
                            s.ticker AS Ticker, 
                            p.date AS Date, 
                            p.open AS Open, 
                            p.high AS High, 
                            p.low AS Low, 
                            p.close AS Close, 
                            p.volume AS Volume
                        ORDER BY p.date
                    '''

            financial_data_query = '''
                        MATCH (stock:Stock_Ticker {ticker: $ticker})
                        OPTIONAL MATCH (stock)-[:HAS_FINANCIAL_STATEMENT_FOR_YEAR]->(year:Year)
                        OPTIONAL MATCH (year)-[:HAS_BALANCE_SHEET]->(balance:Balance_Sheet)
                        OPTIONAL MATCH (year)-[:HAS_INCOME_STATEMENT]->(income:Income_Statement)
                        OPTIONAL MATCH (year)-[:HAS_CASH_FLOW_STATEMENT]->(cashflow:Cash_Flow)
                        RETURN stock, COLLECT(year) AS years, COLLECT(balance) AS balances, COLLECT(income) AS incomes, COLLECT(cashflow) AS cashflows
                    '''

            news_data_query = '''
                        MATCH (stock:Stock_Ticker {ticker: $ticker})
                        OPTIONAL MATCH (stock)-[:HAS_DAILY_PRICE]->(price:DailyPrice)
                        OPTIONAL MATCH (price)-[:HAS_NEWS]->(news:News)
                        RETURN stock,
                                COLLECT(price) AS prices,
                                COLLECT(news {.*, sentiment: news.sentiment, sentiment_score: news.sentiment_score }) AS newsListWithPrice
                    '''

            buy_sell_signal_query = '''
                        MATCH (s:Stock_Ticker {ticker: $ticker})-[:HAS_DAILY_PRICE]->(p:DailyPrice)
                        WHERE p.date >= date() - duration({days: 90})
                        OPTIONAL MATCH (p)-[:NEXT]->(next:DailyPrice)
                        RETURN p.date AS CurrentDate, p.close AS CurrentClose, 
                               next.date AS NextDate, next.close AS NextClose,
                               CASE 
                                   WHEN next.close IS NOT NULL AND next.close > p.close THEN "Buy"
                                   WHEN next.close IS NOT NULL AND next.close < p.close THEN "Sell"
                                   ELSE "Hold"
                               END AS Signal
                        ORDER BY p.date
                    '''

            stock_data = fetch_data(stock_data_query, ticker)
            financial_data = fetch_data(financial_data_query, ticker)
            news_data = fetch_data(news_data_query, ticker)
            buy_sell_signal_data = fetch_data(buy_sell_signal_query, ticker)

            if not stock_data:
                print(f"No stock data found for {ticker}. Skipping...")
                continue

            if not financial_data:
                print(f"No financial data found for {ticker}. Skipping...")
                continue

            if not news_data:
                print(f"No news data found for {ticker}. Skipping...")
                continue

            if not buy_sell_signal_data:
                print(f"No buy/sell signal data found for {ticker}. Skipping...")
                return

            # Convert to DataFrame
            buy_sell_signal_df = pd.DataFrame(buy_sell_signal_data)

            # Ensure date is sorted and converted to datetime
            buy_sell_signal_df['CurrentDate'] = pd.to_datetime(buy_sell_signal_df['CurrentDate'])
            buy_sell_signal_df.sort_values('CurrentDate', inplace=True)

            # Plot stock prices
            fig = plt.figure(figsize=(12, 6))
            plt.plot(buy_sell_signal_df['CurrentDate'], buy_sell_signal_df['CurrentClose'], label='Stock Price',
                     marker='o', color='blue',
                     markersize=4, linewidth=0.5)

            # Add Buy/Sell markers
            buy_signals = buy_sell_signal_df[buy_sell_signal_df['Signal'] == 'Buy']
            sell_signals = buy_sell_signal_df[buy_sell_signal_df['Signal'] == 'Sell']

            # Plot buy signals
            plt.scatter(buy_signals['CurrentDate'], buy_signals['CurrentClose'],
                        color='green', label='Buy Signal', marker='^', s=60)

            # Plot sell signals
            plt.scatter(sell_signals['CurrentDate'], sell_signals['CurrentClose'],
                        color='red', label='Sell Signal', marker='v', s=60)

            # Customize the plot
            plt.title(f"Stock Price with Buy/Sell Signals for {ticker}")
            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.legend()
            plt.grid()
            st.pyplot(fig)

            # Answer user's question
            answer = answer_user_question(user_question, stock_data, financial_data, news_data)
            # print(f"\nAnalysis for {ticker}:\n{answer}")
            all_answers.append(f"Ticker: {ticker}\n{answer}")

        # Combine all answers for multiple tickers
        if all_answers:
            print("\nSummary of analyses:")
            summary_answer = summary_analysis("\n".join(all_answers), user_question)
            #print(summary_answer)
            return summary_answer
        else:
            print("No data available for any of the tickers provided.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return