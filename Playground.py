import openai
import os
import phi
import phi.api


from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.playground import Playground, serve_playground_app

#load Environment Variables from .env file
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")


# Websearch Agent
websearch_agent = Agent(
    name ='websearch_agent',
    role = " Sear the web for financial information",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions = ["Always include the sources"],
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
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True,
)

MultiModelAgent = Agent(
    team=[websearch_agent, finance_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    name="Multi-Model Agent",
    instructions=["Always include sources and use tables to display data where possible."],
    show_tool_calls=True,
    markdown=True,
)


app = Playground(agents=[websearch_agent, finance_agent, MultiModelAgent]).get_app()

if __name__ == "__main__":
    serve_playground_app("Playground:app", reload = True)