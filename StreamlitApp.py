import streamlit as st
import openai
import InvestmentDecisionAssistant
import StockAnalysis
import FinancialNewsAnalysis
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")

def get_question_type(question):
    messages = [
        {"role": "system",
         "content": "You are a financial assistant that determines the type of financial information the user is asking about. Your response should only be one of the following: 'Investment-related', 'Stock-related' or 'Financial_or_news-related'" 
                    "For questions like should I buy or should I sell the company it should be 'Investment-related'."},
        {"role": "user",
         "content": f"What type of financial information is being requested in this question: {question}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

def get_answer(question, question_type):
    answer = []
    try:
        if question_type == "Investment-related":
            answer = InvestmentDecisionAssistant.invest_decision_main(question)
            return answer
        elif question_type == "Stock-related":
            answer = StockAnalysis.stock_analysis_main(question)
        elif question_type == "Financial_or_news-related":
            answer = FinancialNewsAnalysis.financial_news_analysis_main(question)
    except Exception as e:
        return f"Error: {e}"
    return answer

# Streamlit application
def main():
    # Application Title
    st.title("💹 Investment Decision Assistant")
    st.markdown(
        """
        <center><h5>Welcome! </h5>
        <body>Here to provide insightful answers to help you make an investment decision for the right company!
        </body></center><br>
        """,
    unsafe_allow_html=True
    )

    # Initialize the session state for the loop
    if "exit_triggered" not in st.session_state:
        st.session_state["exit_triggered"] = False

    user_question = st.text_input("Enter your question:")
    question_type = get_question_type(user_question)
    if user_question:
        if user_question.lower() == "exit":
            st.session_state["exit_triggered"] = True
            st.write("Goodbye!")
        else:
            with st.spinner("Thinking..."):
                if question_type == "Investment-related":
                    answer = get_answer(user_question, question_type)
                    st.write(answer)
                else:
                    all_answer = get_answer(user_question, question_type)
                    for answer in all_answer:
                        st.write(answer)

    if st.button("Exit"):
        st.write("Thank you for using the Investment Decision Assistant.")
        st.write("Exiting application...")
        st.stop()

if __name__ == "__main__":
    main()