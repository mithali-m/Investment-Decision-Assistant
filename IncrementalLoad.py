import FetchData
from neo4j import GraphDatabase
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j Driver
driver = GraphDatabase.driver(uri=NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def stock_data_incremental_load(stock_data, tickers):
    # Function to get last fetch date from Neo4j
    def get_last_fetch_date(stock_ticker):
        with driver.session() as session:
            result = session.run(f"MATCH (t:Stock_Ticker {{ticker: $stock_ticker}}) RETURN t.last_fetch_date", stock_ticker = stock_ticker)
            last_date = result.single().get("t.last_fetch_date")
            return last_date.iso_format() if last_date else "2000-01-01"  # Default to a very old date if no data exists

    # Function to update the last fetch date in Neo4j
    def update_last_fetch_date(stock_ticker, last_date):
        with driver.session() as session:
            session.run(f"MATCH (t:Stock_Ticker {{ticker: $stock_ticker}}) SET t.last_fetch_date = date($last_date)", stock_ticker = stock_ticker, last_date = last_date)

    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.strftime('%Y-%m-%d')

    transaction_execution_commands = []

    # Loop through each unique ticker
    for ticker in tickers:
        # Step 1: Get the last fetch date for the current ticker from Neo4j
        last_fetch_date = get_last_fetch_date(ticker)
        # Step 2: Filter data for the current ticker and only fetch data after the last fetch date
        ticker_data = stock_data[(stock_data['Ticker'] == ticker) & (stock_data['Date'] > last_fetch_date)]

        if ticker_data.empty:
            print(f"No new data for ticker: {ticker}")
            continue  # Skip if there's no new data for this ticker

        # Step 3: Get the most recent date in the filtered data
        latest_date = ticker_data['Date'].max()

        # Step 4: Create the list of daily prices in Cypher-compatible format
        daily_prices = ', '.join([
            f"{{date: date('{row['Date']}'), open: {row['Open']}, high: {row['High']}, low: {row['Low']}, close: {row['Close']}, volume: {row['Volume']}, change: {row['Change']}, change_percent: {row['Change Percent']}}}"
            for _, row in ticker_data.iterrows()
        ])

        # Step 5: Prepare the Cypher query for incremental data
        neo4j_create_statement = f'''
            MERGE (stock:Stock_Ticker {{ticker: "{ticker}"}})
            WITH stock
            UNWIND [{daily_prices}] AS dailyPrice
            MERGE (price:DailyPrice {{date: dailyPrice.date, ticker: "{ticker}"}})
            SET price.open = dailyPrice.open,
                price.high = dailyPrice.high,
                price.low = dailyPrice.low,
                price.close = dailyPrice.close,
                price.volume = dailyPrice.volume,
                price.change = dailyPrice.change,
                price.change_percent = dailyPrice.change_percent
            MERGE (stock)-[:HAS_DAILY_PRICE]->(price);
            '''
        # Append each ticker's query to the list
        transaction_execution_commands.append(neo4j_create_statement)

        # Debug: print the query being executed for each ticker
        print(f"Executing query for ticker: {ticker}")

        # Step 6: Update the last fetch date for the current ticker in Neo4j
        update_last_fetch_date(ticker, latest_date)

    # Function to execute the Cypher queries
    def execute_transactions(transaction_execution_command):
        # Connect to Neo4j
        session = driver.session()

        # Execute each query
        for command in transaction_execution_command:
            session.run(command)

        # Close session
        session.close()

    # Execute the generated Cypher queries
    execute_transactions(transaction_execution_commands)

    # Create next relationship
    next_relationship_commands = []

    for ticker in tickers:
        neo4j_next_statement = f'''
        MATCH (stock:Stock_Ticker {{ticker: "{ticker}"}})-[:HAS_DAILY_PRICE]->(price:DailyPrice)
        WITH price
        ORDER BY price.date
        WITH collect(price) AS prices
        UNWIND RANGE(0, size(prices) - 2) AS i
        WITH prices[i] AS current, prices[i + 1] AS next
        MERGE (current)-[:NEXT]->(next);
        '''

        next_relationship_commands.append(neo4j_next_statement)

    # Execute the commands to create NEXT relationships
    execute_transactions(next_relationship_commands)
    return

def news_data_incremental_load(news_data):
    # Function to check if a news entry already exists
    def news_exists(ticker, date_of_news, headline, description, source, url):
        with driver.session() as session:
            result = session.run('''
            MATCH (news:News {ticker: $ticker, date_of_news: date($date_of_news), headline: $headline, description: $description, source: $source, url: $url})
            RETURN COUNT(news) AS count
            ''', ticker=ticker, date_of_news=date_of_news, headline=headline, description=description, source=source, url=url)
            return result.single()['count'] > 0

    # Function to add news for a ticker and connect it to the DailyPrice node in Neo4j
    def add_news_to_daily_price(ticker, headline, description, source, date_of_news, url, sentiment, sentiment_score):
        with driver.session() as session:
            session.run('''
            MATCH (stock:Stock_Ticker {ticker: $ticker})
            OPTIONAL MATCH (stock)-[:HAS_DAILY_PRICE]->(price:DailyPrice {date: date($date_of_news)})
            WITH stock, price
            // If there is a DailyPrice match, connect news to it
            FOREACH (_ IN CASE WHEN price IS NOT NULL THEN [1] ELSE [] END |
                MERGE (news:News {
                    ticker: $ticker,
                    headline: $headline,
                    description: $description,
                    source: $source,
                    date_of_news: date($date_of_news),
                    url: $url,
                    sentiment: $sentiment,
                    sentiment_score: $sentiment_score
                })
                MERGE (price)-[:HAS_NEWS]->(news)
            )
            // If no DailyPrice exists for the news date, create a Date node and connect news to it
            FOREACH (_ IN CASE WHEN price IS NULL THEN [1] ELSE [] END |
                MERGE (news:News {
                    ticker: $ticker,
                    headline: $headline,
                    description: $description,
                    source: $source,
                    date_of_news: date($date_of_news),
                    url: $url,
                    sentiment: $sentiment,
                    sentiment_score: $sentiment_score
                })
                MERGE (dateNode:Date {date: date($date_of_news), ticker: $ticker})
                MERGE (stock)-[:HAS_NO_DAILY_PRICE]->(dateNode)
                MERGE (dateNode)-[:HAS_NEWS]->(news)
            );
            ''',
            ticker=ticker,
            headline=headline,
            description=description,
            source=source,
            date_of_news=date_of_news,
            url=url,
            sentiment=sentiment,
            sentiment_score=sentiment_score)

    # Function to perform incremental load
    def incremental_load(news):
        news['Date of News'] = pd.to_datetime(news['Date of News']).dt.strftime('%Y-%m-%d')

        for index, row in news.iterrows():
            # Skip if the news already exists in the database
            if news_exists(row['Ticker'], row['Date of News'], row['Headline'], row['Description'], row['Source'], row['URL']):
                # print(f"Skipping existing news: {row['Headline']} for ticker {row['Ticker']} on {row['Date of News']}")
                continue

            print(f"Adding news: {row['Headline']} for ticker {row['Ticker']} on {row['Date of News']}")
            # Add the news if it doesn't exist
            add_news_to_daily_price(
                ticker=row['Ticker'],
                headline=row['Headline'],
                description=row['Description'],
                source=row['Source'],
                date_of_news=row['Date of News'],
                url=row['URL'],
                sentiment=row['Sentiment'],
                sentiment_score=row['Sentiment_Score']
            )

    # Incremental load execution
    incremental_load(news_data)
    return

def main():
    tickers = ['AAPL', 'NVDA', 'AMD', 'JPM', 'BX', 'AMZN', 'TSLA', 'SBUX', 'JNJ', 'MRK']
    stock_data = FetchData.get_incremental_stock_data(tickers)
    news_data = FetchData.get_incremental_news_data(tickers)
    stock_data_incremental_load(stock_data, tickers)
    news_data_incremental_load(news_data)

if __name__=="__main__":
    main()