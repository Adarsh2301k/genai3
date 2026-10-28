import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")

search_tool = TavilySearch(max_results=5)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
)
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant that answers questions based on the following news context:\n{news}"
)
chain = prompt | llm | StrOutputParser()

news = search_tool.invoke({"query": "What is the latest news about AI?"})
print(chain.invoke({"news": news}))
