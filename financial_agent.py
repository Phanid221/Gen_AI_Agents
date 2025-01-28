from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Websearch Agent
websearch_agent = Agent(
    name ='websearch_agent',
    role = " Sear the web for financial information",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions = ["Always include the sources"],
    show_tool_calls = True,
    markdown = 
    True,
)



#Financial Agent
finance_agent = Agent(
    name = "Finance AI Agent",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, 
                      company_news=True),
        ],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True,
)


#Multi-Model-Agent

MultiModelAgent = Agent(
    team=[websearch_agent, finance_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    name="Multi-Model Agent",
    instructions=["Always include sources and use tables to display data where possible."],
    show_tool_calls=True,
    markdown=True,
)

MultiModelAgent.print_response("Summarize latest news on Nvidia and provide the stock price and analyst recommendations with the lates news.", stream=True)

