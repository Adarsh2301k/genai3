import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")

search_tool = TavilySearchResults(max_results=5)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant that answers questions based on the following news context:\n{news}"
)
chain = prompt | llm | StrOutputParser()

news = search_tool.run("What is the latest news about AI?")
print(chain.invoke({"news": news}))
