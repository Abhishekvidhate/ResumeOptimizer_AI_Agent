from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_PROJECT = ''  # this means "default" is selected as project
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_TRACING_V2 = "true"
print("GROQ_API_KEY")
print("hello world")

llm = ChatGroq(model_name="llama3-8b-8192")
response = llm.invoke("convert into marathi language , text : my name is abhishek")
print(response)